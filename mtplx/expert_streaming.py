"""Policy primitives for SSD-backed routed-expert slot banks.

This module deliberately has no MLX dependency.  It owns the cache decision
that sits between a resident MoE router and a future native Metal slot-bank
loader.  Keeping the policy pure makes route traces reproducible and lets us
size a cache before allocating multi-gigabyte Metal buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


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


@dataclass(frozen=True)
class SlotEviction:
    """A persistent slot reassigned to a hotter expert."""

    slot: int
    previous_expert: int
    next_expert: int


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
        if expert_record_bytes < 0:
            raise ValueError("expert_record_bytes must be non-negative")
        self.route_calls += 1
        self.expert_requests += len(plan.hits) + len(plan.misses)
        self.expert_hits += len(plan.hits)
        self.expert_misses += len(plan.misses)
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
    ) -> None:
        if expert_count <= 0:
            raise ValueError("expert_count must be positive")
        if persistent_slots < 0:
            raise ValueError("persistent_slots must be non-negative")
        if transient_slots <= 0:
            raise ValueError("transient_slots must be positive")
        if not 0.0 < frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")

        self.expert_count = expert_count
        self.persistent_slots = persistent_slots
        self.transient_slots = transient_slots
        self.slot_count = persistent_slots + transient_slots
        self.frequency_decay = frequency_decay

        self._epoch = 0
        self._slot_to_expert: list[int | None] = [None] * self.persistent_slots
        self._expert_to_slot: dict[int, int] = {}
        self._history = [_ExpertHistory() for _ in range(expert_count)]

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return tuple(expert for expert in self._slot_to_expert if expert is not None)

    @property
    def occupancy(self) -> int:
        return len(self._expert_to_slot)

    def _validate_experts(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        experts = tuple(int(expert) for expert in expert_ids)
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
        age = self._epoch - history.score_epoch
        if age <= 0 or history.score == 0.0:
            return history.score
        return history.score * (self.frequency_decay**age)

    def _touch_decode(self, expert: int) -> None:
        history = self._history[expert]
        history.score = self._score(expert) + 1.0
        history.score_epoch = self._epoch
        history.last_used = self._epoch

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
        self._epoch += 1
        unique_experts = tuple(dict.fromkeys(experts))

        if phase is RoutingPhase.DECODE:
            for expert in unique_experts:
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
            if phase is RoutingPhase.DECODE and self.persistent_slots:
                persistent_slot = self._empty_persistent_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned=pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_expert[victim_slot]
                        assert victim is not None
                        if self._score(expert) > self._score(victim):
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

        for expert in hit_set:
            self._history[expert].last_used = self._epoch

        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=tuple(miss_order),
            loads=tuple(loads),
            evictions=tuple(evictions),
        )


@dataclass
class ExpertCacheSimulation:
    """Multi-layer cache simulator and aggregate I/O accounting."""

    expert_count: int
    persistent_slots: int
    transient_slots: int
    expert_record_bytes: int
    frequency_decay: float = 0.995
    counters: CacheCounters = field(default_factory=CacheCounters)
    _layers: dict[int, LayerExpertSlotBank] = field(default_factory=dict)

    def layer(self, layer_index: int) -> LayerExpertSlotBank:
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if layer_index not in self._layers:
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
        if effective_ssd_bytes_per_second <= 0:
            raise ValueError("effective_ssd_bytes_per_second must be positive")
        counters = self.counters.as_dict()
        return {
            **counters,
            "estimated_io_seconds": self.counters.bytes_read
            / effective_ssd_bytes_per_second,
            "layers_observed": len(self._layers),
            "persistent_cache_bytes": len(self._layers)
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
