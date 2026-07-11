"""Policy primitives for SSD-backed routed-expert slot banks.

This module deliberately has no MLX dependency.  It owns the cache decision
that sits between a resident MoE router and a future native Metal slot-bank
loader.  Keeping the policy pure makes route traces reproducible and lets us
size a cache before allocating multi-gigabyte Metal buffers.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from operator import index
from typing import Iterable


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


class RoutingPhase(str, Enum):
    """Inference phase used to select the admission policy."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class SlotLoad:
    """One expert record that must be loaded before dispatch."""

    expert: int
    slot: int
    persistent: bool
    generation: int | None = None


@dataclass(frozen=True)
class SlotEviction:
    """A persistent slot reassigned to a hotter expert."""

    slot: int
    previous_expert: int
    next_expert: int
    previous_layer: int | None = None
    next_layer: int | None = None


@dataclass(frozen=True)
class RoutePlan:
    """Resolved slot mapping for one layer invocation.

    ``slots`` preserves the router's input order.  A native executor can use
    it in place of the original expert ids after all ``loads`` have completed.
    """

    phase: RoutingPhase
    experts: tuple[int, ...]
    slots: tuple[int, ...]
    hits: tuple[int, ...]
    misses: tuple[int, ...]
    loads: tuple[SlotLoad, ...]
    evictions: tuple[SlotEviction, ...]
    generations: tuple[int | None, ...] = ()


@dataclass
class CacheCounters:
    """Aggregate counters suitable for CLI/server metrics."""

    route_calls: int = 0
    expert_requests: int = 0
    expert_hits: int = 0
    expert_misses: int = 0
    persistent_loads: int = 0
    transient_loads: int = 0
    evictions: int = 0
    bytes_read: int = 0

    def observe(self, plan: RoutePlan, *, expert_record_bytes: int) -> None:
        expert_record_bytes = _integer(
            "expert_record_bytes", expert_record_bytes, minimum=0
        )
        hit_experts = set(plan.hits)
        assignment_hits = sum(expert in hit_experts for expert in plan.experts)
        self.route_calls += 1
        self.expert_requests += len(plan.experts)
        self.expert_hits += assignment_hits
        self.expert_misses += len(plan.experts) - assignment_hits
        self.persistent_loads += sum(load.persistent for load in plan.loads)
        self.transient_loads += sum(not load.persistent for load in plan.loads)
        self.evictions += len(plan.evictions)
        self.bytes_read += len(plan.loads) * expert_record_bytes

    @property
    def hit_rate(self) -> float:
        total = self.expert_hits + self.expert_misses
        return self.expert_hits / total if total else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "route_calls": self.route_calls,
            "expert_requests": self.expert_requests,
            "expert_hits": self.expert_hits,
            "expert_misses": self.expert_misses,
            "hit_rate": self.hit_rate,
            "persistent_loads": self.persistent_loads,
            "transient_loads": self.transient_loads,
            "evictions": self.evictions,
            "bytes_read": self.bytes_read,
        }


@dataclass
class _ExpertHistory:
    score: float = 0.0
    score_epoch: int = 0
    last_used: int = -1


@dataclass
class _GlobalDirectoryEntry:
    slot: int
    generation: int
    state: str


class LayerExpertSlotBank:
    """Per-layer TinyLFU-like hot bank with transient service slots.

    The persistent tier learns only from decode routes.  Prefill misses use the
    transient tier and therefore cannot evict a useful decode hot set.  Decode
    misses are admitted only once their decayed frequency beats the coldest
    unpinned resident; otherwise they are served through a transient slot.

    This class plans I/O but never performs it.  The future native layer owns
    fixed Metal buffers and applies the returned ``SlotLoad`` operations using
    aligned ``pread``.
    """

    def __init__(
        self,
        *,
        expert_count: int,
        persistent_slots: int,
        transient_slots: int,
        frequency_decay: float = 0.995,
        cache_policy: str = "frequency",
    ) -> None:
        expert_count = _integer("expert_count", expert_count, minimum=1)
        persistent_slots = _integer("persistent_slots", persistent_slots, minimum=0)
        transient_slots = _integer("transient_slots", transient_slots, minimum=1)
        if persistent_slots > expert_count:
            raise ValueError("persistent_slots cannot exceed expert_count")
        if isinstance(frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            frequency_decay = float(frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(frequency_decay) or not 0.0 < frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")

        self.expert_count = expert_count
        self.persistent_slots = persistent_slots
        self.transient_slots = transient_slots
        self.slot_count = persistent_slots + transient_slots
        self.frequency_decay = frequency_decay
        if cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        self.cache_policy = cache_policy

        # Prefill traffic must neither age nor refresh decode admission state.
        # A separate decode-only epoch keeps a long prompt from erasing the
        # hot set immediately before generation starts.
        self._decode_epoch = 0
        self._slot_to_expert: list[int | None] = [None] * self.persistent_slots
        self._expert_to_slot: dict[int, int] = {}
        self._history = [_ExpertHistory() for _ in range(expert_count)]
        self._prefill_seed_candidates: set[int] = set()

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return tuple(expert for expert in self._slot_to_expert if expert is not None)

    @property
    def occupancy(self) -> int:
        return len(self._expert_to_slot)

    def invalidate_expert(self, expert_id: int) -> int | None:
        """Forget a failed/stale persistent mapping and return its slot."""

        expert = _integer("expert id", expert_id, minimum=0)
        if expert >= self.expert_count:
            raise ValueError(
                f"expert id {expert} is outside [0, {self.expert_count})"
            )
        slot = self._expert_to_slot.pop(expert, None)
        if slot is not None:
            self._slot_to_expert[slot] = None
        return slot

    def reset(self) -> None:
        """Clear residency and decode history without changing capacity."""

        self._decode_epoch = 0
        self._slot_to_expert = [None] * self.persistent_slots
        self._expert_to_slot.clear()
        self._history = [_ExpertHistory() for _ in range(self.expert_count)]
        self._prefill_seed_candidates.clear()

    def prepare_prefill_seed(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        """Choose prompt-frequent experts for empty slots without eviction."""

        empty = self.persistent_slots - self.occupancy
        if empty <= 0:
            self._prefill_seed_candidates.clear()
            return ()
        experts = self._validate_experts_for_seed(expert_ids)
        counts = Counter(experts)
        ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))
        chosen = tuple(
            expert
            for expert in ranked
            if expert not in self._expert_to_slot
        )[:empty]
        self._prefill_seed_candidates = set(chosen)
        return chosen

    def _validate_experts_for_seed(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        for expert in experts:
            if expert >= self.expert_count:
                raise ValueError(
                    f"expert id {expert} is outside [0, {self.expert_count})"
                )
        return experts

    def _validate_experts(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        if not experts:
            raise ValueError("a route must select at least one expert")
        for expert in experts:
            if not 0 <= expert < self.expert_count:
                raise ValueError(
                    f"expert id {expert} is outside [0, {self.expert_count})"
                )
        unique_count = len(dict.fromkeys(experts))
        if unique_count > self.transient_slots:
            raise ValueError(
                "transient_slots must cover the maximum unique experts in one route"
            )
        return experts

    def _score(self, expert: int) -> float:
        history = self._history[expert]
        age = self._decode_epoch - history.score_epoch
        if age <= 0 or history.score == 0.0:
            return history.score
        return history.score * (self.frequency_decay**age)

    def _touch_decode(self, expert: int) -> None:
        history = self._history[expert]
        history.score = self._score(expert) + 1.0
        history.score_epoch = self._decode_epoch
        history.last_used = self._decode_epoch

    def _empty_persistent_slot(self) -> int | None:
        for slot, expert in enumerate(self._slot_to_expert):
            if expert is None:
                return slot
        return None

    def _victim_slot(self, *, pinned: set[int]) -> int | None:
        candidates: list[tuple[float, int, int]] = []
        for slot, expert in enumerate(self._slot_to_expert):
            if expert is None or expert in pinned:
                continue
            history = self._history[expert]
            if self.cache_policy == "lru":
                candidates.append((float(history.last_used), 0, slot))
            else:
                candidates.append((self._score(expert), history.last_used, slot))
        return min(candidates)[2] if candidates else None

    def _assign_persistent(
        self,
        *,
        slot: int,
        expert: int,
        evictions: list[SlotEviction],
    ) -> None:
        previous = self._slot_to_expert[slot]
        if previous is not None:
            del self._expert_to_slot[previous]
            evictions.append(
                SlotEviction(
                    slot=slot,
                    previous_expert=previous,
                    next_expert=expert,
                )
            )
        self._slot_to_expert[slot] = expert
        self._expert_to_slot[expert] = slot

    def plan(
        self,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        """Resolve router expert ids to persistent or transient slots."""

        experts = self._validate_experts(expert_ids)
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))

        if phase is RoutingPhase.DECODE:
            self._decode_epoch += 1
            for expert in experts:
                self._touch_decode(expert)

        hit_set = {
            expert for expert in unique_experts if expert in self._expert_to_slot
        }
        miss_order = [expert for expert in unique_experts if expert not in hit_set]
        resolved: dict[int, int] = {
            expert: self._expert_to_slot[expert] for expert in hit_set
        }
        loads: list[SlotLoad] = []
        evictions: list[SlotEviction] = []
        pinned = set(hit_set)
        transient_experts: list[int] = []

        for expert in miss_order:
            persistent_slot: int | None = None
            if phase is RoutingPhase.PREFILL and expert in self._prefill_seed_candidates:
                persistent_slot = self._empty_persistent_slot()
                self._prefill_seed_candidates.discard(expert)
            elif phase is RoutingPhase.DECODE and self.persistent_slots:
                persistent_slot = self._empty_persistent_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned=pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_expert[victim_slot]
                        assert victim is not None
                        # Do not let a first-seen singleton evict a resident
                        # merely because decay made the resident score < 1.
                        # A second decode observation (or older history) must
                        # first lift the candidate above this admission floor.
                        if self.cache_policy == "lru" or self._score(expert) > max(
                            1.0, self._score(victim)
                        ):
                            persistent_slot = victim_slot

            if persistent_slot is None:
                transient_experts.append(expert)
                continue

            self._assign_persistent(
                slot=persistent_slot,
                expert=expert,
                evictions=evictions,
            )
            pinned.add(expert)
            resolved[expert] = persistent_slot
            loads.append(SlotLoad(expert=expert, slot=persistent_slot, persistent=True))

        transient_base = self.persistent_slots
        for offset, expert in enumerate(transient_experts):
            slot = transient_base + offset
            resolved[expert] = slot
            loads.append(SlotLoad(expert=expert, slot=slot, persistent=False))

        if phase is RoutingPhase.DECODE:
            for expert in hit_set:
                self._history[expert].last_used = self._decode_epoch

        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=tuple(miss_order),
            loads=tuple(loads),
            evictions=tuple(evictions),
        )


class GlobalExpertSlotBank:
    """One fixed expert-record cache shared by every routed model layer.

    Keys include both layer and expert ID because equal expert IDs in different
    layers name unrelated weights.  The physical buffers are allocated once;
    this policy only changes the generation-safe indirection from a key to a
    global slot.  Prefill initially admits at most the legacy uniform quota per
    layer so early layers cannot consume the entire empty pool.  Decode then
    allows the replacement policy to move capacity between layers.
    """

    def __init__(
        self,
        *,
        layer_indices: Iterable[int],
        expert_count: int,
        persistent_slots: int,
        transient_slots: int,
        prefill_slots_per_layer: int,
        frequency_decay: float = 0.995,
        cache_policy: str = "lru",
    ) -> None:
        self.layer_indices = tuple(
            _integer("layer index", layer, minimum=0) for layer in layer_indices
        )
        if not self.layer_indices or len(set(self.layer_indices)) != len(
            self.layer_indices
        ):
            raise ValueError("layer_indices must contain unique routed layers")
        self._layer_set = set(self.layer_indices)
        self.expert_count = _integer("expert_count", expert_count, minimum=1)
        self.persistent_slots = _integer(
            "persistent_slots", persistent_slots, minimum=0
        )
        self.transient_slots = _integer(
            "transient_slots", transient_slots, minimum=1
        )
        self.prefill_slots_per_layer = _integer(
            "prefill_slots_per_layer", prefill_slots_per_layer, minimum=0
        )
        if self.prefill_slots_per_layer > self.expert_count:
            raise ValueError("prefill_slots_per_layer cannot exceed expert_count")
        maximum_keys = len(self.layer_indices) * self.expert_count
        if self.persistent_slots > maximum_keys:
            raise ValueError("persistent_slots cannot exceed routed expert count")
        if isinstance(frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            self.frequency_decay = float(frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(self.frequency_decay) or not 0.0 < self.frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        if cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        self.cache_policy = cache_policy
        self.slot_count = self.persistent_slots + self.transient_slots

        self._decode_epoch = 0
        self._slot_to_key: list[tuple[int, int] | None] = [
            None
        ] * self.persistent_slots
        self._key_to_slot: dict[tuple[int, int], int] = {}
        self._directory: dict[tuple[int, int], _GlobalDirectoryEntry] = {}
        self._slot_generations: list[int] = [0] * self.persistent_slots
        self._free_slots = deque(range(self.persistent_slots))
        self._free_slot_set = set(range(self.persistent_slots))
        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._history: dict[tuple[int, int], _ExpertHistory] = {}
        self._layer_occupancy: Counter[int] = Counter()
        self._evictions = 0
        self._cross_layer_evictions = 0
        self._prefill_seed_candidates: dict[int, set[int]] = {
            layer: set() for layer in self.layer_indices
        }

    @property
    def occupancy(self) -> int:
        return len(self._key_to_slot)

    @property
    def resident_experts_by_layer(self) -> dict[int, tuple[int, ...]]:
        grouped: dict[int, list[int]] = {layer: [] for layer in self.layer_indices}
        for key in self._slot_to_key:
            if key is not None:
                grouped[key[0]].append(key[1])
        return {layer: tuple(experts) for layer, experts in grouped.items()}

    @property
    def occupancy_by_layer(self) -> dict[int, int]:
        return {layer: int(self._layer_occupancy[layer]) for layer in self.layer_indices}

    def _key(self, layer: int, expert: int) -> tuple[int, int]:
        layer = _integer("layer index", layer, minimum=0)
        expert = _integer("expert id", expert, minimum=0)
        if layer not in self._layer_set:
            raise ValueError(f"layer {layer} is not a routed model layer")
        if expert >= self.expert_count:
            raise ValueError(
                f"expert id {expert} is outside [0, {self.expert_count})"
            )
        return layer, expert

    def _validate_experts(
        self, layer: int, expert_ids: Iterable[int]
    ) -> tuple[int, tuple[int, ...]]:
        layer = self._key(layer, 0)[0]
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        if not experts:
            raise ValueError("a route must select at least one expert")
        for expert in experts:
            self._key(layer, expert)
        if len(dict.fromkeys(experts)) > self.transient_slots:
            raise ValueError(
                "transient_slots must cover the maximum unique experts in one route"
            )
        return layer, experts

    def _history_for(self, key: tuple[int, int]) -> _ExpertHistory:
        history = self._history.get(key)
        if history is None:
            history = _ExpertHistory()
            self._history[key] = history
        return history

    def _score(self, key: tuple[int, int]) -> float:
        history = self._history_for(key)
        age = self._decode_epoch - history.score_epoch
        if age <= 0 or history.score == 0.0:
            return history.score
        return history.score * (self.frequency_decay**age)

    def _touch_decode(self, key: tuple[int, int]) -> None:
        history = self._history_for(key)
        history.score = self._score(key) + 1.0
        history.score_epoch = self._decode_epoch
        history.last_used = self._decode_epoch

    def _empty_slot(self) -> int | None:
        if not self._free_slots:
            return None
        slot = self._free_slots.popleft()
        self._free_slot_set.remove(slot)
        return slot

    def _victim_slot(self, *, pinned: set[tuple[int, int]]) -> int | None:
        if self.cache_policy == "lru":
            for key, slot in self._lru.items():
                if key not in pinned:
                    return slot
            return None
        candidates: list[tuple[float, int, int]] = []
        for slot, key in enumerate(self._slot_to_key):
            if key is None or key in pinned:
                continue
            history = self._history_for(key)
            if self.cache_policy == "lru":
                candidates.append((float(history.last_used), 0, slot))
            else:
                candidates.append((self._score(key), history.last_used, slot))
        return min(candidates)[2] if candidates else None

    def _assign(
        self,
        *,
        slot: int,
        key: tuple[int, int],
        evictions: list[SlotEviction],
    ) -> None:
        previous = self._slot_to_key[slot]
        if previous is not None:
            del self._key_to_slot[previous]
            self._directory.pop(previous, None)
            self._lru.pop(previous, None)
            self._layer_occupancy[previous[0]] -= 1
            self._evictions += 1
            if previous[0] != key[0]:
                self._cross_layer_evictions += 1
            evictions.append(
                SlotEviction(
                    slot=slot,
                    previous_expert=previous[1],
                    next_expert=key[1],
                    previous_layer=previous[0],
                    next_layer=key[0],
                )
            )
        elif slot in self._free_slot_set:
            # Defensive support for callers assigning a specifically chosen
            # empty slot rather than consuming it via _empty_slot().
            self._free_slot_set.remove(slot)
            self._free_slots.remove(slot)
        self._slot_generations[slot] += 1
        self._slot_to_key[slot] = key
        self._key_to_slot[key] = slot
        self._directory[key] = _GlobalDirectoryEntry(
            slot=slot,
            generation=self._slot_generations[slot],
            state="loading",
        )
        self._lru[key] = slot
        self._lru.move_to_end(key)
        self._layer_occupancy[key[0]] += 1

    def prepare_prefill_seed(
        self, layer: int, expert_ids: Iterable[int]
    ) -> tuple[int, ...]:
        layer, experts = self._validate_experts(layer, expert_ids)
        remaining_layer = max(
            0, self.prefill_slots_per_layer - self._layer_occupancy[layer]
        )
        empty = self.persistent_slots - self.occupancy
        available = min(remaining_layer, empty)
        if available <= 0:
            self._prefill_seed_candidates[layer].clear()
            return ()
        counts = Counter(experts)
        ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))
        chosen = tuple(
            expert
            for expert in ranked
            if (layer, expert) not in self._key_to_slot
        )[:available]
        self._prefill_seed_candidates[layer] = set(chosen)
        return chosen

    def invalidate_expert(self, layer: int, expert_id: int) -> int | None:
        key = self._key(layer, expert_id)
        slot = self._key_to_slot.pop(key, None)
        if slot is not None:
            self._slot_to_key[slot] = None
            self._directory.pop(key, None)
            self._lru.pop(key, None)
            self._layer_occupancy[layer] -= 1
            if slot not in self._free_slot_set:
                self._free_slots.append(slot)
                self._free_slot_set.add(slot)
        return slot

    def publish_ready(self, layer: int, plan: RoutePlan) -> None:
        """Publish successfully filled global generations as cache hits."""

        for load in plan.loads:
            if not load.persistent or load.generation is None:
                continue
            key = self._key(layer, load.expert)
            entry = self._directory.get(key)
            if (
                entry is None
                or entry.slot != load.slot
                or entry.generation != load.generation
                or entry.state != "loading"
            ):
                raise RuntimeError("global cache generation changed before publish")
            entry.state = "ready"

    def rollback(self, layer: int, plan: RoutePlan) -> tuple[tuple[int, int], ...]:
        """Remove only loading directory entries reserved by this plan."""

        removed: list[tuple[int, int]] = []
        for load in plan.loads:
            if not load.persistent or load.generation is None:
                continue
            key = self._key(layer, load.expert)
            entry = self._directory.get(key)
            if (
                entry is None
                or entry.slot != load.slot
                or entry.generation != load.generation
                or entry.state != "loading"
            ):
                continue
            del self._directory[key]
            self._key_to_slot.pop(key, None)
            self._lru.pop(key, None)
            if self._slot_to_key[load.slot] == key:
                self._slot_to_key[load.slot] = None
                self._layer_occupancy[layer] -= 1
                if load.slot not in self._free_slot_set:
                    self._free_slots.append(load.slot)
                    self._free_slot_set.add(load.slot)
            removed.append((load.slot, load.generation))
        return tuple(removed)

    def plan(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        layer, experts = self._validate_experts(layer, expert_ids)
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))
        keys = tuple((layer, expert) for expert in unique_experts)

        if phase is RoutingPhase.DECODE:
            self._decode_epoch += 1
            for expert in experts:
                self._touch_decode((layer, expert))

        hit_keys = {
            key
            for key in keys
            if (entry := self._directory.get(key)) is not None
            and entry.state == "ready"
        }
        if self.cache_policy == "lru":
            for key in keys:
                if key in hit_keys:
                    self._lru.move_to_end(key)
        hit_set = {expert for key_layer, expert in hit_keys if key_layer == layer}
        miss_order = [expert for expert in unique_experts if expert not in hit_set]
        resolved = {
            expert: self._key_to_slot[(layer, expert)] for expert in hit_set
        }
        loads: list[SlotLoad] = []
        evictions: list[SlotEviction] = []
        pinned = set(hit_keys)
        transient_experts: list[int] = []

        for expert in miss_order:
            key = (layer, expert)
            persistent_slot: int | None = None
            if (
                phase is RoutingPhase.PREFILL
                and expert in self._prefill_seed_candidates[layer]
                and self._layer_occupancy[layer] < self.prefill_slots_per_layer
            ):
                persistent_slot = self._empty_slot()
                self._prefill_seed_candidates[layer].discard(expert)
            elif phase is RoutingPhase.DECODE and self.persistent_slots:
                persistent_slot = self._empty_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned=pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_key[victim_slot]
                        assert victim is not None
                        if self.cache_policy == "lru" or self._score(key) > max(
                            1.0, self._score(victim)
                        ):
                            persistent_slot = victim_slot

            if persistent_slot is None:
                transient_experts.append(expert)
                continue
            self._assign(slot=persistent_slot, key=key, evictions=evictions)
            pinned.add(key)
            resolved[expert] = persistent_slot
            loads.append(
                SlotLoad(
                    expert=expert,
                    slot=persistent_slot,
                    persistent=True,
                    generation=self._directory[key].generation,
                )
            )

        transient_base = self.persistent_slots
        for offset, expert in enumerate(transient_experts):
            slot = transient_base + offset
            resolved[expert] = slot
            loads.append(SlotLoad(expert=expert, slot=slot, persistent=False))

        if phase is RoutingPhase.DECODE:
            for key in hit_keys:
                self._history_for(key).last_used = self._decode_epoch

        generations = tuple(
            (
                self._directory[(layer, expert)].generation
                if resolved[expert] < self.persistent_slots
                else None
            )
            for expert in experts
        )
        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=tuple(miss_order),
            loads=tuple(loads),
            evictions=tuple(evictions),
            generations=generations,
        )

    def reset(self) -> None:
        self._decode_epoch = 0
        self._slot_to_key = [None] * self.persistent_slots
        self._key_to_slot.clear()
        self._directory.clear()
        self._slot_generations = [0] * self.persistent_slots
        self._free_slots = deque(range(self.persistent_slots))
        self._free_slot_set = set(range(self.persistent_slots))
        self._lru.clear()
        self._history.clear()
        self._layer_occupancy.clear()
        self._evictions = 0
        self._cross_layer_evictions = 0
        for candidates in self._prefill_seed_candidates.values():
            candidates.clear()

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity": self.persistent_slots,
            "occupancy": self.occupancy,
            "occupancy_by_layer": self.occupancy_by_layer,
            "evictions": self._evictions,
            "cross_layer_evictions": self._cross_layer_evictions,
            # Records are read directly into their final fixed slots. There is
            # no live-weight compaction or relocation copy on this path.
            "relocation_bytes": 0,
        }


@dataclass
class ExpertCacheSimulation:
    """Multi-layer cache simulator and aggregate I/O accounting."""

    expert_count: int
    persistent_slots: int
    transient_slots: int
    expert_record_bytes: int
    allocated_layer_count: int | None = None
    frequency_decay: float = 0.995
    counters: CacheCounters = field(default_factory=CacheCounters)
    _layers: dict[int, LayerExpertSlotBank] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expert_count = _integer("expert_count", self.expert_count, minimum=1)
        self.persistent_slots = _integer(
            "persistent_slots", self.persistent_slots, minimum=0
        )
        self.transient_slots = _integer(
            "transient_slots", self.transient_slots, minimum=1
        )
        self.expert_record_bytes = _integer(
            "expert_record_bytes", self.expert_record_bytes, minimum=1
        )
        if self.persistent_slots > self.expert_count:
            raise ValueError("persistent_slots cannot exceed expert_count")
        if self.allocated_layer_count is not None:
            self.allocated_layer_count = _integer(
                "allocated_layer_count", self.allocated_layer_count, minimum=1
            )
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            self.frequency_decay = float(self.frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(self.frequency_decay) or not 0.0 < self.frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")

    def layer(self, layer_index: int) -> LayerExpertSlotBank:
        layer_index = _integer("layer_index", layer_index, minimum=0)
        if layer_index not in self._layers:
            if (
                self.allocated_layer_count is not None
                and len(self._layers) >= self.allocated_layer_count
            ):
                raise ValueError("trace exceeds allocated_layer_count")
            self._layers[layer_index] = LayerExpertSlotBank(
                expert_count=self.expert_count,
                persistent_slots=self.persistent_slots,
                transient_slots=self.transient_slots,
                frequency_decay=self.frequency_decay,
            )
        return self._layers[layer_index]

    def observe(
        self,
        *,
        layer_index: int,
        expert_ids: Iterable[int],
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        plan = self.layer(layer_index).plan(expert_ids, phase=phase)
        self.counters.observe(plan, expert_record_bytes=self.expert_record_bytes)
        return plan

    def summary(self, *, effective_ssd_bytes_per_second: float) -> dict[str, object]:
        if isinstance(effective_ssd_bytes_per_second, bool):
            raise TypeError("effective_ssd_bytes_per_second must be finite")
        try:
            effective_ssd_bytes_per_second = float(effective_ssd_bytes_per_second)
        except (TypeError, ValueError) as exc:
            raise TypeError("effective_ssd_bytes_per_second must be finite") from exc
        if (
            not isfinite(effective_ssd_bytes_per_second)
            or effective_ssd_bytes_per_second <= 0
        ):
            raise ValueError("effective_ssd_bytes_per_second must be positive")
        counters = self.counters.as_dict()
        layer_count = (
            len(self._layers)
            if self.allocated_layer_count is None
            else self.allocated_layer_count
        )
        return {
            **counters,
            "estimated_io_seconds": self.counters.bytes_read
            / effective_ssd_bytes_per_second,
            "layers_observed": len(self._layers),
            "allocated_layer_count": layer_count,
            "persistent_cache_scope": (
                "observed_layers_only"
                if self.allocated_layer_count is None
                else "configured_model"
            ),
            "persistent_cache_bytes": layer_count
            * self.persistent_slots
            * self.expert_record_bytes,
            "observed_layer_cache_bytes": len(self._layers)
            * self.persistent_slots
            * self.expert_record_bytes,
            # Native execution reuses one top-k scratch bank across sequential
            # layers; it is intentionally not multiplied by layer count.
            "transient_scratch_bytes": self.transient_slots * self.expert_record_bytes,
            "resident_experts_by_layer": {
                str(layer): list(bank.resident_experts)
                for layer, bank in sorted(self._layers.items())
            },
        }
