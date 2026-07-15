from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.expert_streaming import RoutingPhase
from mtplx.generation import (
    _clear_cache_every,
    _defer_verify_hidden_eval_enabled,
    _make_target_prefill_cache,
    _maybe_repage_target_prefill_cache,
    _prefill,
    _prefill_cache_only_forward,
    _prefill_chunk_cache_cleanup_every,
    _prefill_chunk_size,
    _prefill_committed_mtp_history_streaming,
    _prefill_with_hidden_sequence,
    _sustained_prefill_layout,
    generate_ar,
    generate_mtpk,
    restore_or_prefill_prompt_state,
)
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.vision.splice import VisionSplice


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        position_offset=None,
    ):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        self.calls.append(
            {
                "tokens": int(input_ids.shape[1]),
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class KwargsOnlyTinyModel(TinyModel):
    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        **kwargs,
    ):
        return super().__call__(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )


class AcceptingTinyMTPModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        length = int(next_token_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        logits = mx.zeros((1, length, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class OffsetCache:
    def __init__(self):
        self.offset = 0
        self.trimmed: list[int] = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(self.offset), int(n))
        self.offset -= n
        self.trimmed.append(n)
        return n


class CompilableOffsetCache(OffsetCache):
    step = 64

    def __init__(self):
        super().__init__()
        self.keys = mx.zeros((1, 1, 64, 1), dtype=mx.float32)
        self.values = mx.zeros((1, 1, 64, 1), dtype=mx.float32)

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        start = mx.array(self.offset, dtype=mx.int32)
        self.keys = mx.slice_update(self.keys, keys, start, axes=(2,))
        self.values = mx.slice_update(self.values, values, start, axes=(2,))
        self.offset += steps
        return self.keys, self.values


class RejectingTinyMTPModel(AcceptingTinyMTPModel):
    def __init__(self):
        super().__init__()
        self.target_cache = [OffsetCache()]

    def make_cache(self):
        return self.target_cache

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if cache:
            for entry in cache:
                entry.offset += int(input_ids.shape[1])
        return super().__call__(input_ids, cache=cache, **kwargs)

    def mtp_forward(self, *args, **kwargs):
        result = super().mtp_forward(*args, **kwargs)
        if isinstance(result, tuple):
            logits, hidden = result
            logits = mx.zeros_like(logits) + mx.array(
                [0.0, 0.0, 1.0, 0.0],
                dtype=mx.float32,
            )
            return logits, hidden
        return mx.zeros_like(result) + mx.array(
            [0.0, 0.0, 1.0, 0.0],
            dtype=mx.float32,
        )


class CycleTrackingTinyMTPModel(AcceptingTinyMTPModel):
    def __init__(
        self,
        *,
        draft_token: int = 1,
        target_verify_token: int = 1,
        fail_on_draft: int | None = None,
    ):
        super().__init__()
        self.draft_token = int(draft_token)
        self.target_verify_token = int(target_verify_token)
        self.fail_on_draft = fail_on_draft
        self.cycle_active = False
        self.draft_calls = 0
        self.finish_calls: list[object] = []

    def make_mtp_cache(self):
        return [OffsetCache()]

    def finish_mtp_cycle(self, mtp_cache):
        self.finish_calls.append(mtp_cache)
        self.cycle_active = False

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if self.cycle_active:
            raise AssertionError("target verification observed active MTP cycle")
        result = super().__call__(input_ids, cache=cache, **kwargs)
        if int(input_ids.shape[1]) <= 1:
            return result
        token_logits = mx.full((4,), -1.0, dtype=mx.float32)
        token_logits[self.target_verify_token] = 1.0
        if isinstance(result, tuple):
            logits, hidden = result
            return mx.zeros_like(logits) + token_logits, hidden
        return mx.zeros_like(result) + token_logits

    def mtp_forward(self, hidden_states, next_token_ids, **kwargs):
        self.draft_calls += 1
        self.cycle_active = True
        if self.fail_on_draft == self.draft_calls:
            raise RuntimeError("synthetic recurrent draft failure")
        length = int(next_token_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        token_logits = mx.full((4,), -1.0, dtype=mx.float32)
        token_logits[self.draft_token] = 1.0
        logits = mx.zeros((1, length, 4), dtype=mx.float32) + token_logits
        if kwargs.get("return_hidden", False):
            return logits, hidden
        return logits


class VisionTinyMTPModel(CycleTrackingTinyMTPModel):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            embed_tokens=lambda token_ids: mx.zeros(
                (*token_ids.shape, 2),
                dtype=mx.float32,
            )
        )
        self.input_embeddings: list[mx.array | None] = []

    def __call__(self, input_ids, *, input_embeddings=None, cache=None, **kwargs):
        self.input_embeddings.append(input_embeddings)
        return super().__call__(input_ids, cache=cache, **kwargs)


class CompilableCycleTrackingTinyMTPModel(CycleTrackingTinyMTPModel):
    def make_mtp_cache(self):
        return [CompilableOffsetCache()]

    @staticmethod
    def _append_cache(mtp_cache, next_token_ids) -> None:
        if mtp_cache is None:
            return
        values = next_token_ids.astype(mx.float32).reshape(
            1,
            1,
            int(next_token_ids.shape[1]),
            1,
        )
        mtp_cache[0].update_and_fetch(values, values + 100.0)

    def mtp_forward(self, hidden_states, next_token_ids, **kwargs):
        result = super().mtp_forward(hidden_states, next_token_ids, **kwargs)
        self._append_cache(kwargs.get("mtp_cache"), next_token_ids)
        return result

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        **_kwargs,
    ):
        self._append_cache(mtp_cache, next_token_ids)
        return hidden_states


class CompilableRejectAtSecondDepthTinyMTPModel(CompilableCycleTrackingTinyMTPModel):
    def __call__(self, input_ids, *, cache=None, **kwargs):
        if self.cycle_active:
            raise AssertionError("target verification observed active MTP cycle")
        result = TinyModel.__call__(self, input_ids, cache=cache, **kwargs)
        row_tokens = [int(token) for row in input_ids.tolist() for token in row]
        logits = mx.full((1, len(row_tokens), 4), -100.0, dtype=mx.float32)
        for index, token in enumerate(row_tokens):
            logits[0, index, (token + 1) % 4] = 100.0
        if isinstance(result, tuple):
            _unused_logits, hidden = result
            return logits, hidden
        return logits

    def mtp_forward(self, hidden_states, next_token_ids, **kwargs):
        self.draft_calls += 1
        self.cycle_active = True
        depth = int(kwargs.get("mtp_depth", 1))
        draft_token = (3, 2, 1)[min(depth - 1, 2)]
        hidden = mx.zeros((1, 1, 2), dtype=mx.float32)
        logits = mx.full((1, 1, 4), -100.0, dtype=mx.float32)
        logits[0, 0, draft_token] = 100.0
        self._append_cache(kwargs.get("mtp_cache"), next_token_ids)
        if kwargs.get("return_hidden", False):
            return logits, hidden
        return logits


class StopAfterFirstDraftPolicy:
    current_depth = 3
    wants_draft_metrics = False

    def should_continue_after_draft(self, **_kwargs):
        return {"continue": False, "reason": "synthetic_early_exit"}

    def observe(self, **_kwargs):
        return {"action": "hold", "next_depth": self.current_depth}


class RejectAtSecondDepthTinyMTPModel(CycleTrackingTinyMTPModel):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if self.cycle_active:
            raise AssertionError("target verification observed active MTP cycle")
        result = TinyModel.__call__(self, input_ids, cache=cache, **kwargs)
        row_tokens = [int(token) for row in input_ids.tolist() for token in row]
        logits = mx.full((1, len(row_tokens), 4), -100.0, dtype=mx.float32)
        for index, token in enumerate(row_tokens):
            logits[0, index, (token + 1) % 4] = 100.0
        if isinstance(result, tuple):
            _unused_logits, hidden = result
            return logits, hidden
        return logits

    def mtp_forward(self, hidden_states, next_token_ids, **kwargs):
        self.draft_calls += 1
        self.cycle_active = True
        draft_tokens = (2, 0, 1)
        draft_token = draft_tokens[min(self.draft_calls - 1, len(draft_tokens) - 1)]
        hidden = mx.zeros((1, 1, 2), dtype=mx.float32)
        logits = mx.full((1, 1, 4), -100.0, dtype=mx.float32)
        logits[0, 0, draft_token] = 100.0
        if kwargs.get("return_hidden", False):
            return logits, hidden
        return logits


class PendingBonusThenRejectTinyMTPModel(CycleTrackingTinyMTPModel):
    def __init__(self):
        super().__init__()
        self.target_cache = [OffsetCache()]
        self.committed_mtp_cache: list[OffsetCache] | None = None
        self.cycle_base: int | None = None

    @property
    def target_cache_offset(self) -> int:
        return self.target_cache[0].offset

    @property
    def mtp_committed_offset(self) -> int:
        assert self.committed_mtp_cache is not None
        return self.committed_mtp_cache[0].offset

    def make_cache(self):
        return self.target_cache

    def make_mtp_cache(self):
        self.committed_mtp_cache = [OffsetCache()]
        return self.committed_mtp_cache

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if self.cycle_active:
            raise AssertionError("target verification observed active MTP cycle")
        if cache:
            for entry in cache:
                entry.offset += int(input_ids.shape[1])
        result = TinyModel.__call__(self, input_ids, cache=cache, **kwargs)
        row_tokens = [int(token) for row in input_ids.tolist() for token in row]
        logits = mx.full((1, len(row_tokens), 8), -100.0, dtype=mx.float32)
        for index, token in enumerate(row_tokens):
            logits[0, index, (token + 1) % 8] = 100.0
        if isinstance(result, tuple):
            _unused_logits, hidden = result
            return logits, hidden
        return logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        **kwargs,
    ):
        self.draft_calls += 1
        if not self.cycle_active:
            self.cycle_base = mtp_cache[0].offset if mtp_cache else 0
        self.cycle_active = True
        source_token = int(next_token_ids.item())
        draft_token = 0 if self.draft_calls == 4 else (source_token + 1) % 8
        if mtp_cache:
            for entry in mtp_cache:
                entry.offset += 1
        hidden = mx.zeros((1, 1, 2), dtype=mx.float32)
        logits = mx.full((1, 1, 8), -100.0, dtype=mx.float32)
        logits[0, 0, draft_token] = 100.0
        if kwargs.get("return_hidden", False):
            return logits, hidden
        return logits

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        **_kwargs,
    ):
        if mtp_cache:
            for entry in mtp_cache:
                entry.offset += int(next_token_ids.shape[1])
        return hidden_states

    def finish_mtp_cycle(self, mtp_cache):
        super().finish_mtp_cycle(mtp_cache)
        if mtp_cache and self.cycle_base is not None:
            target = self.cycle_base + 1
            for entry in mtp_cache:
                entry.trim(max(0, entry.offset - target))
        self.cycle_base = None


class BatchShapeDriftTinyMTPModel(PendingBonusThenRejectTinyMTPModel):
    """Expose a verifier correction that differs from qlen=1 replay."""

    verifier_correction = 6

    def __call__(self, input_ids, *, cache=None, **kwargs):
        result = super().__call__(input_ids, cache=cache, **kwargs)
        if int(input_ids.shape[1]) <= 1:
            return result

        if isinstance(result, tuple):
            logits, hidden = result
            logits = mx.array(logits)
            logits[:, 0, :] = -100.0
            logits[:, 0, self.verifier_correction] = 100.0
            return logits, hidden

        logits = mx.array(result)
        logits[:, 0, :] = -100.0
        logits[:, 0, self.verifier_correction] = 100.0
        return logits


def _runtime(model: TinyModel, *, mtp_enabled: bool = True) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )


def _run_cycle_tracking_mtpk(
    model: CycleTrackingTinyMTPModel,
    *,
    max_tokens: int = 2,
    speculative_depth: int = 1,
    stop_token_ids: set[int] | None = None,
    adaptive_policy=None,
):
    return generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=speculative_depth,
        mtp_history_policy="cycle",
        verify_strategy="batched",
        stop_token_ids=set() if stop_token_ids is None else stop_token_ids,
        adaptive_policy=adaptive_policy,
    )


def test_generate_mtpk_rejects_fresh_recurrent_cache_before_prefill():
    model = AcceptingTinyMTPModel()
    model.mtp_recurrent_requires_persistent_cache = True

    with pytest.raises(ValueError, match="persistent cache"):
        generate_mtpk(
            _runtime(model, mtp_enabled=True),
            [0],
            max_tokens=3,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
            speculative_depth=2,
            mtp_cache_policy="fresh",
            mtp_history_policy="cycle",
            verify_strategy="batched",
            stop_token_ids=set(),
        )

    assert model.calls == []


def test_generate_mtpk_cycle_cleanup_precedes_rejection_verify():
    model = CycleTrackingTinyMTPModel(draft_token=2, target_verify_token=1)

    output = _run_cycle_tracking_mtpk(model)

    assert output.stats.rejected_drafts == 1
    assert len(model.finish_calls) == 1
    assert model.cycle_active is False


def test_generate_mtpk_cycle_cleanup_precedes_accepted_stop_verify():
    model = CycleTrackingTinyMTPModel(draft_token=2, target_verify_token=2)

    output = _run_cycle_tracking_mtpk(
        model,
        max_tokens=3,
        speculative_depth=2,
        stop_token_ids={2},
    )

    assert output.tokens == [1, 2]
    assert len(model.finish_calls) == 1
    assert model.cycle_active is False


def test_generate_mtpk_cycle_cleanup_precedes_adaptive_early_exit_verify():
    model = CycleTrackingTinyMTPModel()

    output = _run_cycle_tracking_mtpk(
        model,
        max_tokens=3,
        speculative_depth=3,
        adaptive_policy=StopAfterFirstDraftPolicy(),
    )

    assert output.stats.events[0]["gated_stop_depth"] == 1
    assert len(model.finish_calls) == 1
    assert model.cycle_active is False


def test_generate_mtpk_cycle_cleanup_runs_when_recurrent_draft_raises():
    model = CycleTrackingTinyMTPModel(fail_on_draft=1)

    with pytest.raises(RuntimeError, match="synthetic recurrent draft failure"):
        _run_cycle_tracking_mtpk(model)

    assert len(model.finish_calls) == 1
    assert model.cycle_active is False


def test_generate_mtpk_evaluated_by_depth_stops_at_first_rejection():
    model = RejectAtSecondDepthTinyMTPModel()

    output = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="cycle",
        verify_strategy="batched",
        stop_token_ids={3},
    )

    assert output.tokens == [1, 2, 3]
    assert output.stats.drafted_by_depth == [1, 1, 1]
    assert output.stats.evaluated_by_depth == [1, 1, 0]
    assert output.stats.accepted_by_depth == [1, 0, 0]
    assert output.stats.evaluated_drafts == 2
    assert output.stats.fully_accepted_verify_calls == 0
    assert output.stats.mean_accept_probability_by_depth == [1.0, 0.0, None]


def test_generate_mtpk_counts_a_fully_accepted_verify_call():
    model = CycleTrackingTinyMTPModel()

    output = _run_cycle_tracking_mtpk(
        model,
        max_tokens=4,
        speculative_depth=3,
    )

    assert output.stats.evaluated_by_depth == [1, 1, 1]
    assert output.stats.evaluated_drafts == 3
    assert output.stats.fully_accepted_verify_calls == 1


@pytest.mark.parametrize("depth", range(1, 8))
def test_device_k_draft_preserves_serial_tokens_and_verification(
    monkeypatch,
    depth: int,
):
    import mtplx.generation as generation

    monkeypatch.setattr(
        generation.mx,
        "compile",
        lambda fn: fn,
    )
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=4)
    stock_runtime = _runtime(
        CompilableCycleTrackingTinyMTPModel(),
        mtp_enabled=True,
    )
    stock = generate_mtpk(
        stock_runtime,
        [0, 1],
        max_tokens=depth + 1,
        sampler=sampler,
        speculative_depth=depth,
        mtp_history_policy="committed",
        verify_strategy="batched",
        draft_core="stock",
        stop_token_ids=set(),
        capture_final_state=True,
    )
    compiled_runtime = _runtime(
        CompilableCycleTrackingTinyMTPModel(),
        mtp_enabled=True,
    )
    compiled = generate_mtpk(
        compiled_runtime,
        [0, 1],
        max_tokens=depth + 1,
        sampler=sampler,
        speculative_depth=depth,
        mtp_history_policy="committed",
        verify_strategy="batched",
        draft_core="device-k",
        stop_token_ids=set(),
        capture_final_state=True,
    )

    assert compiled.tokens == stock.tokens
    assert compiled.stats.accepted_by_depth == stock.stats.accepted_by_depth
    assert compiled.stats.evaluated_by_depth == stock.stats.evaluated_by_depth
    assert compiled.stats.verify_calls == stock.stats.verify_calls
    trace_fields = ("depth", "token", "accepted", "accept_probability", "correction")
    assert [
        {name: draft.get(name) for name in trace_fields}
        for draft in compiled.stats.events[0]["drafts"]
    ] == [
        {name: draft.get(name) for name in trace_fields}
        for draft in stock.stats.events[0]["drafts"]
    ]
    assert (
        compiled.stats.events[0]["accepted_depths"]
        == stock.stats.events[0]["accepted_depths"]
    )
    assert compiled.stats.events[0].get("rejected_at_depth") == stock.stats.events[
        0
    ].get("rejected_at_depth")
    assert compiled.stats.events[0]["verify_strategy"] == "batched"
    assert compiled.stats.draft_core["selected"] == "device-k"
    assert compiled.stats.draft_core["per_depth"][str(depth)]["calls"] == 1
    assert compiled_runtime.diagnostic_counters["make_mtp_cache_calls"] == 1
    assert compiled.stats.draft_core["fallbacks"] == 0
    assert compiled.stats.draft_core["organic_compile_calls"] == 0
    assert compiled.stats.draft_core["per_depth"][str(depth)]["live_cache_commits"] == 1
    assert compiled.stats.draft_core["per_depth"][str(depth)]["host_syncs"] == 0
    assert (
        compiled.stats.draft_core["per_depth"][str(depth)]["host_token_transfers"] == 0
    )
    assert (
        compiled.stats.draft_core["per_depth"][str(depth)]["device_handoff_calls"] == 1
    )
    assert (
        compiled.stats.draft_core["per_depth"][str(depth)]["acceptance_host_transfers"]
        == 1
    )
    assert (
        compiled.stats.draft_core["per_depth"][str(depth)]["per_row_argmax_host_reads"]
        == 0
    )
    assert compiled.stats.events[0]["draft_device_handoff"]["enabled"] is True
    assert compiled.stats.events[0]["draft_device_handoff"]["proposal_rows"] == depth
    assert compiled.stats.events[0]["draft_device_handoff"]["verify_rows"] == depth + 1
    assert compiled.stats.draft_core["device_handoff"] == {
        "schema": "compiled-mtp-device-handoff-v1",
        "full_verify_width": True,
        "calls": 1,
        "fallbacks": 0,
        "fallback_reasons": {},
        "acceptance_host_transfers": 1,
        "acceptance_payload_ints": depth + 3,
        "per_row_argmax_host_reads": 0,
        "qualification_eligible": True,
    }
    assert compiled.stats.events[0]["draft_core_dispatch"][
        "primary_optimized_depth"
    ] is (depth == 3)
    prewarm_time = compiled.stats.draft_core["prewarm_time_s"]
    assert prewarm_time > 0.0
    assert compiled.stats.decode_elapsed_s == pytest.approx(
        compiled.stats.elapsed_s - compiled.stats.prompt_eval_time_s - prewarm_time
    )
    assert stock.final_state is not None
    assert compiled.final_state is not None
    stock_cache = stock.final_state.final_committed_mtp_cache[0]
    compiled_cache = compiled.final_state.final_committed_mtp_cache[0]
    mx.eval(stock_cache.keys, stock_cache.values)
    mx.eval(compiled_cache.keys, compiled_cache.values)
    assert compiled_cache.offset == stock_cache.offset
    assert mx.array_equal(compiled_cache.keys, stock_cache.keys).item()
    assert mx.array_equal(compiled_cache.values, stock_cache.values).item()


def test_device_k_prewarm_failure_falls_back_without_dispatch(monkeypatch):
    import mtplx.generation as generation

    class FailingPrewarmBank:
        def __init__(self):
            self.run_calls = 0

        def prewarm(self, *_args, **_kwargs):
            raise RuntimeError("synthetic compile failure")

        def run(self, *_args, **_kwargs):
            self.run_calls += 1
            raise AssertionError("failed prewarm must never enter measured dispatch")

    bank = FailingPrewarmBank()
    monkeypatch.setattr(
        generation,
        "_get_compiled_mtp_draft_bank",
        lambda *_args, **_kwargs: bank,
    )

    output = generate_mtpk(
        _runtime(CompilableCycleTrackingTinyMTPModel(), mtp_enabled=True),
        [0, 1],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        draft_core="device-k",
        stop_token_ids=set(),
    )

    assert bank.run_calls == 0
    assert output.stats.draft_core["selected"] == "stock"
    assert output.stats.draft_core["fallback_reasons"] == {"prewarm:RuntimeError": 1}


def test_device_k_handoff_setup_failure_falls_back_before_state_commit(monkeypatch):
    import mtplx.generation as generation

    monkeypatch.setattr(generation.mx, "compile", lambda fn: fn)

    def fail_before_commit(self, *_args, **_kwargs):
        del self
        raise RuntimeError("synthetic device handoff setup failure")

    monkeypatch.setattr(
        generation.CompiledMTPDraftBank,
        "run_device",
        fail_before_commit,
    )

    output = generate_mtpk(
        _runtime(CompilableCycleTrackingTinyMTPModel(), mtp_enabled=True),
        [0, 1],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        draft_core="device-k",
        stop_token_ids=set(),
    )

    assert output.tokens == [1, 1, 1, 1]
    assert output.stats.draft_core["selected"] == "stock"
    assert output.stats.draft_core["fallbacks"] == 1
    assert output.stats.draft_core["fallback_reasons"] == {"dispatch:RuntimeError": 1}
    assert output.stats.draft_core["device_handoff"]["fallbacks"] == 1
    assert output.stats.draft_core["device_handoff"]["fallback_reasons"] == {
        "dispatch:RuntimeError": 1
    }
    assert output.stats.events[0]["draft_core_fallback"]["reason"] == (
        "dispatch:RuntimeError"
    )


def test_device_k_full_width_handoff_preserves_lazy_bonus_host_path(monkeypatch):
    import mtplx.generation as generation

    monkeypatch.setattr(generation.mx, "compile", lambda fn: fn)
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")

    output = generate_mtpk(
        _runtime(CompilableCycleTrackingTinyMTPModel(), mtp_enabled=True),
        [0, 1],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        draft_core="device-k",
        stop_token_ids=set(),
    )

    event = output.stats.events[0]
    depth = output.stats.draft_core["per_depth"]["3"]
    assert event["draft_device_handoff"] == {
        "enabled": False,
        "reason": "lazy_bonus_verify",
        "proposal_rows": 3,
        "verify_rows": 4,
    }
    assert event["lazy_bonus_verify"]["enabled"] is True
    assert event["lazy_bonus_verify"]["verify_input_tokens"] == 3
    assert depth["host_syncs"] == 1
    assert depth["host_token_transfers"] == 1
    assert depth["device_handoff_calls"] == 0
    assert output.stats.draft_core["device_handoff"]["qualification_eligible"] is False
    assert output.stats.draft_core["device_handoff"]["fallbacks"] == 1
    assert output.stats.draft_core["device_handoff"]["fallback_reasons"] == {
        "lazy_bonus_verify": 1
    }


def test_device_k_committed_history_reject_rolls_back_exact_live_cache(monkeypatch):
    import mtplx.generation as generation

    monkeypatch.setattr(generation.mx, "compile", lambda fn: fn)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=4)

    def run(draft_core: str):
        return generate_mtpk(
            _runtime(
                CompilableRejectAtSecondDepthTinyMTPModel(),
                mtp_enabled=True,
            ),
            [0, 1],
            max_tokens=4,
            sampler=sampler,
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="batched",
            draft_core=draft_core,
            stop_token_ids={0},
            capture_final_state=True,
        )

    stock = run("stock")
    compiled = run("device-k")

    assert compiled.tokens == stock.tokens == [2, 3, 0]
    assert compiled.stats.accepted_by_depth == stock.stats.accepted_by_depth
    assert compiled.stats.accepted_by_depth == [1, 0, 0]
    assert compiled.stats.draft_core["fallbacks"] == 0
    assert compiled.final_state is not None
    assert stock.final_state is not None
    stock_cache = stock.final_state.final_committed_mtp_cache[0]
    compiled_cache = compiled.final_state.final_committed_mtp_cache[0]
    mx.eval(stock_cache.keys, stock_cache.values)
    mx.eval(compiled_cache.keys, compiled_cache.values)
    assert compiled_cache.offset == stock_cache.offset
    assert mx.array_equal(compiled_cache.keys, stock_cache.keys).item()
    assert mx.array_equal(compiled_cache.values, stock_cache.values).item()


def test_cold_committed_history_reports_exact_rows_and_tps(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "0")
    model = CycleTrackingTinyMTPModel()
    appended: list[tuple[list[int], int]] = []

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        **_kwargs,
    ):
        appended.append((list(token_ids), int(hidden_states.shape[1])))
        return 2.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    output = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0, 1, 2, 3],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=1,
        mtp_history_policy="committed",
        stop_token_ids=set(),
    )

    assert appended == [([1, 2, 3], 3)]
    assert output.stats.new_prefill_tokens == 4
    assert output.stats.prompt_mtp_history_tokens == 3
    assert output.stats.prompt_mtp_history_time_s == 2.0
    assert output.stats.prompt_mtp_history_tok_s == pytest.approx(1.5)
    assert [call["tokens"] for call in model.calls] == [3, 1]
    assert [call["return_hidden"] for call in model.calls] == [True, True]


@pytest.mark.parametrize("prompt_ids", [[7], [0, 7]], ids=["only", "final"])
def test_prefill_with_hidden_sequence_splices_final_vision_token(prompt_ids):
    model = VisionTinyMTPModel()
    vision_row = mx.array([[9.0, 10.0]], dtype=mx.float32)
    splice = VisionSplice(
        image_pad_token_id=7,
        embeddings=vision_row,
    )

    _prefill_with_hidden_sequence(
        _runtime(model, mtp_enabled=True),
        prompt_ids,
        hidden_variant="post_norm",
        vision_splice=splice,
    )

    final_embeddings = model.input_embeddings[-1]
    assert final_embeddings is not None
    assert mx.array_equal(final_embeddings, vision_row[None, :, :]).item()
    assert splice.remaining() == 0


@pytest.mark.parametrize(
    ("stop_token_ids", "expected_finish_reason"),
    [(set(), "length"), ({1}, "stop")],
    ids=["length", "stop"],
)
def test_generate_mtpk_final_state_commits_terminal_primary_to_both_caches(
    stop_token_ids: set[int], expected_finish_reason: str
):
    model = PendingBonusThenRejectTinyMTPModel()

    output = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=stop_token_ids,
        capture_final_state=True,
    )

    assert output.tokens == [1]
    assert output.final_state is not None
    assert output.final_state.safe_to_commit is True
    assert output.final_state.finish_reason == expected_finish_reason
    assert output.stats.final_state_capture_time_s > 0.0
    assert model.target_cache_offset == 2
    assert model.mtp_committed_offset == 1

    cold_continuation = generate_ar(
        _runtime(PendingBonusThenRejectTinyMTPModel(), mtp_enabled=False),
        [0, 1],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )
    resumed_token = int(mx.argmax(output.final_state.final_logits[0]).item())
    assert resumed_token == cold_continuation.tokens[0] == 2


def test_generate_mtpk_pending_bonus_then_rejection_matches_ar():
    sampler = SamplerConfig(temperature=0.6, top_p=1.0, top_k=1)
    ar_model = PendingBonusThenRejectTinyMTPModel()
    mtpk_model = PendingBonusThenRejectTinyMTPModel()

    ar = generate_ar(
        _runtime(ar_model, mtp_enabled=False),
        [0],
        max_tokens=6,
        sampler=sampler,
        stop_token_ids=set(),
    )
    mtpk = generate_mtpk(
        _runtime(mtpk_model, mtp_enabled=True),
        [0],
        max_tokens=6,
        sampler=sampler,
        speculative_depth=2,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )

    assert mtpk.tokens == ar.tokens == [1, 2, 3, 4, 5, 6]
    assert mtpk.finish_reason == ar.finish_reason == "length"
    assert mtpk.stats.generated_tokens == ar.stats.generated_tokens
    assert mtpk.stats.events[0]["accepted_depths"] == 2
    assert mtpk.stats.events[0]["bonus_token"] == 4
    assert mtpk.stats.events[1]["accepted_depths"] == 1
    assert mtpk.stats.events[1]["rejected_at_depth"] == 2
    assert mtpk_model.target_cache_offset == 1 + mtpk.stats.generated_tokens
    assert mtpk_model.mtp_committed_offset == mtpk.stats.generated_tokens


def test_generate_mtpk_greedy_rejection_commits_batched_verifier_correction():
    model = BatchShapeDriftTinyMTPModel()

    output = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=8),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )

    rejected = output.stats.events[0]
    correction = rejected["drafts"][0]["correction"]
    assert rejected["rejected_at_depth"] == 1
    assert correction == model.verifier_correction
    assert output.tokens[:2] == [1, correction]
    assert len(output.tokens) == 3
    assert output.final_state is not None
    assert output.final_state.safe_to_commit is True
    assert output.final_state.generated_token_ids == tuple(output.tokens)
    assert model.target_cache_offset == 1 + len(output.tokens)
    assert model.mtp_committed_offset == len(output.tokens)


def test_contiguous_then_repage_cache_layout_restores_paged_env(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(received_cache):
        events.append(("repage", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
        assert received_cache is cache
        return {"enabled": 1, "entries": 0, "skipped": 0}

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed >= 0.0
    assert events == [("make_cache", "0"), ("repage", "1")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_contiguous_dense_decode_cache_layout_does_not_repage(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(_received_cache):
        raise AssertionError("dense decode layout must not repage after prefill")

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed == 0.0
    assert events == [("make_cache", "0")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_session_restore_uses_prefill_layout_cache_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            cache_factory = kwargs["cache_factory"]
            assert callable(cache_factory)
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=cache_factory(),
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "clone"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "clone"


def test_live_frontier_reference_restore_survives_prefill_layout_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            assert callable(kwargs["cache_factory"])
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=["live-frontier-cache"],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="reference_lease",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "reference_lease"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "reference_lease"


def test_auto_sustained_prefill_policy_keeps_dense_decode_through_128k(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY", "auto")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", "16384")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", "256")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    # Dense-layout cleanup cadence: every 4 chunks (2026-07-05 A/B — the
    # per-chunk synchronize+clear_cache cost 5-21% prefill throughput with
    # byte-identical peak memory; receipts in MEASUREMENTS).
    assert _prefill_chunk_cache_cleanup_every() == 4
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 4
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "196608")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_cache_cleanup_every() == 2
    assert _clear_cache_every() == 0


def test_auto_sustained_prefill_policy_repages_when_paged_kv_quant_is_enabled(
    monkeypatch,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q4")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "off")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_non_sustained_long_context_prefill_is_blocked_before_full_hidden_eval(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.delenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    with pytest.raises(
        RuntimeError, match="Blocked unsafe long-context MTP prefill path"
    ):
        restore_or_prefill_prompt_state(
            _runtime(model, mtp_enabled=True),
            list(range(8)),
            mtp_history_policy="committed",
        )

    assert model.calls == []


def test_non_sustained_long_context_prefill_guard_has_explicit_escape_hatch(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", "1")
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    restore_or_prefill_prompt_state(
        _runtime(model, mtp_enabled=True),
        list(range(8)),
        mtp_history_policy="committed",
    )

    assert model.calls


def test_generate_ar_does_not_request_hidden_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_AR_RETURN_HIDDEN", raising=False)
    monkeypatch.delenv("MTPLX_DIAGNOSTIC_AR_RETURN_HIDDEN", raising=False)
    model = TinyModel()

    out = generate_ar(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )

    assert out.stats.ar_return_hidden is False
    assert out.stats.forward_ar_hidden_calls == 0
    assert out.stats.forward_ar_plain_calls >= 1
    assert out.stats.prompt_target_prefill_time_s == out.stats.prompt_eval_time_s
    assert out.stats.prompt_mtp_history_time_s == 0.0
    assert out.stats.prompt_target_prefill_tok_s > 0.0
    assert out.stats.tok_s == out.stats.decode_tok_s
    assert out.stats.decode_elapsed_s == pytest.approx(
        out.stats.elapsed_s - out.stats.prompt_eval_time_s
    )
    assert out.stats.end_to_end_tok_s <= out.stats.decode_tok_s
    assert all(call["return_hidden"] is False for call in model.calls)


def test_lazy_bonus_verify_shortens_full_accept_verify_input(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 3, 1]
    assert out.stats.verify_calls == 1
    assert out.stats.commit_time_s > 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is True
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 3
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 3


def test_lazy_target_distributions_inline_bonus_avoids_bonus_reforward(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 4]
    assert out.stats.verify_calls == 1
    assert out.stats.lazy_bonus_commit_time_s == 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is False
    assert (
        out.stats.events[0]["lazy_bonus_verify"]["disabled_by"]
        == "lazy_target_distributions"
    )
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 4
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})
    assert out.stats.events[0]["target_distribution_materialized"]["mode"] == (
        "lazy_accept_bonus_path"
    )


def test_lazy_target_distributions_stop_after_first_rejection(monkeypatch):
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:1] == [1]
    assert out.stats.events[0]["rejected_at_depth"] == 1
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.target_distribution_materialized_windows == 1
    assert out.stats.events[0]["target_distribution_materialized"]["rows"] == 1


@pytest.mark.parametrize(
    ("model_cls", "sampler"),
    [
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)),
        (RejectingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=1.0, top_k=1)),
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)),
    ],
)
def test_lazy_target_distributions_match_dense_reference(
    monkeypatch,
    model_cls,
    sampler,
):
    def run_once(*, lazy: bool):
        monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
        monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
        monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
        if lazy:
            monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
        else:
            monkeypatch.delenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", raising=False)
        return generate_mtpk(
            _runtime(model_cls(), mtp_enabled=True),
            [0],
            max_tokens=5,
            sampler=sampler,
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
            seed=123,
        )

    dense = run_once(lazy=False)
    lazy = run_once(lazy=True)

    assert lazy.tokens == dense.tokens
    assert lazy.stats.accepted_by_depth == dense.stats.accepted_by_depth
    assert lazy.stats.drafted_by_depth == dense.stats.drafted_by_depth
    assert lazy.stats.rejected_drafts == dense.stats.rejected_drafts
    assert lazy.stats.bonus_tokens == dense.stats.bonus_tokens
    assert lazy.finish_reason == dense.finish_reason


def test_lazy_bonus_verify_skips_d1_by_default(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1, 1]
    lazy = out.stats.events[0]["lazy_bonus_verify"]
    assert lazy["enabled"] is False
    assert lazy["min_depth"] == 2
    assert lazy["verify_input_tokens"] == 2
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})


def test_omit_speculative_bonus_skips_bonus_distribution_row(monkeypatch):
    monkeypatch.setenv("MTPLX_OMIT_SPECULATIVE_BONUS", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.events[0]["speculative_bonus"] == {
        "omitted": True,
        "distribution_row_needed": False,
    }
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 1
    assert "bonus_token" not in out.stats.events[0]
    assert out.stats.bonus_tokens == 0


def test_trim_commit_reuses_rejected_verify_prefix_and_forwards_correction(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=1,
        mtp_history_policy="cycle",
        verify_strategy="trim_commit",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert [call["tokens"] for call in model.calls] == [1, 2, 1]
    assert model.target_cache[0].trimmed == [1]
    assert out.stats.events[0]["capture_repair"] == "trimmed_prefix_correction_forward"
    assert "repair_forward" in out.stats.events[0]["timing_s"]


def test_sustained_prefill_chunks_without_full_prompt_logits(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert model.calls[-1]["logits_keep"] == 1
    assert rt.diagnostic_counters["prefill_chunks"] == 2
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0
    assert rt.diagnostic_counters["final_logits_tokens_emitted"] == 1


def test_warm_restored_suffix_prefill_is_chunked_and_typed_for_abort(monkeypatch):
    # kvcache-v2: suffixes <= MTPLX_SMALL_SUFFIX_FUSED_MAX fuse into one
    # forward; this test guards the chunked lane used above that threshold.
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    prefill_events: list[dict[str, object]] = []

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6],
        mtp_history_policy="committed",
        session_bank=Bank(),
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 3
    assert prompt_state.suffix_tokens == 4
    assert [call["tokens"] for call in model.calls] == [2, 1, 1]
    assert [call["return_hidden"] for call in model.calls] == [True, True, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert appended == [[3], [4, 5], [6]]
    assert rt.diagnostic_counters["restored_suffix_prefill_chunks"] == 2
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    assert [event["tokens_done"] for event in chunk_events] == [3, 5, 6, 7]
    assert [event["tokens_total"] for event in chunk_events] == [7, 7, 7, 7]
    assert [event["cached_tokens"] for event in chunk_events] == [3, 3, 3, 3]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [4, 4, 4, 4]
    assert chunk_events[-1]["live_prefill_tok_s"] is not None


def test_restore_prefers_larger_near_gap_over_shorter_exact_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []

    exact_entry = SimpleNamespace(prefix_len=3)
    near_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-1",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            # kvcache-v2: with boundary-true restore on (default), the
            # block-prefix lane is env-decided (default on) for every
            # client, not just OpenCode-compact — restores fail closed at
            # the entry layer instead (issue #138).
            assert kwargs["allow_block_prefix"] is True
            return [(near_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()
    prefill_events: list[dict[str, object]] = []

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 7
    assert prompt_state.suffix_tokens == 2
    assert prompt_state.restore_mode == "near_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.prefix_restore_calls == [(7, "clone")]
    assert near_entry.hits == 1
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    # kvcache-v2 fused small-suffix prefill emits one progress event for the
    # whole (tiny) suffix instead of per-chunk events.
    assert [event["tokens_done"] for event in chunk_events] == [7, 9]
    assert [event["cached_tokens"] for event in chunk_events] == [7, 7]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [2, 2]


def test_opencode_compact_restore_prefers_block_prefix_over_short_exact(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    opencode_compact_policy = (
        "tool_prompt_mode=compact;"
        "tool_contract=compact_tool_contract:schema_free:v1;"
        "opencode_prompt_contract=opencode_agent"
    )

    exact_entry = SimpleNamespace(prefix_len=3)
    block_entry = SimpleNamespace(
        prefix_len=20,
        token_ids=tuple(range(20)),
        session_id="opencode-session",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=opencode_compact_policy,
        snapshot_epoch=20,
        mtp_snapshot_epoch=20,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []
            self.near_allow_block: list[bool] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_allow_block.append(bool(kwargs["allow_block_prefix"]))
            if not kwargs["allow_block_prefix"]:
                return []
            return [(block_entry, 8)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=opencode_compact_policy,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 8
    assert prompt_state.suffix_tokens == 4
    assert prompt_state.restore_mode == "block_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.near_allow_block == [True]
    assert bank.prefix_restore_calls == [(8, "clone")]
    assert block_entry.hits == 1
    # kvcache-v2 fused small-suffix prefill appends the post-first-token
    # history rows in one call instead of body/final chunks — same rows, same
    # hidden positions, one eval barrier.
    assert appended == [[8], [9, 10, 11]]


def _make_frozen_prefix_bank_fixture(rt, *, policy_fingerprint=None):
    """Bank shape from issue #138: a stale short exact-prefix entry plus a
    much longer entry sharing a bigger prompt prefix (gap > tiny-gap limit).
    Before the fix, non-OpenCode clients could only take the tiny-gap lane,
    so every restore froze on the short exact entry."""
    exact_entry = SimpleNamespace(prefix_len=3)
    block_entry = SimpleNamespace(
        prefix_len=20,
        token_ids=tuple(range(20)),
        session_id="agent-session",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=policy_fingerprint,
        snapshot_epoch=20,
        mtp_snapshot_epoch=20,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []
            self.near_allow_block: list[bool] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_allow_block.append(bool(kwargs["allow_block_prefix"]))
            if not kwargs["allow_block_prefix"]:
                return []
            return [(block_entry, 8)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    return Bank(), exact_entry, block_entry


def _install_history_stub(monkeypatch):
    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)


def test_generic_client_escapes_stale_short_exact_prefix_via_block_restore(
    monkeypatch,
):
    """Issue #138: Pi/little-coder style clients (no OpenCode-compact
    fingerprint) froze on the oldest short exact prefix while longer banked
    prefixes went unused, re-prefilling a growing suffix every turn. With
    boundary-true restore on (the v2 default), the block-prefix lane is safe
    and must engage for every client."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, _exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 8
    assert prompt_state.restore_mode == "block_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.near_allow_block == [True]
    assert bank.prefix_restore_calls == [(8, "clone")]
    assert block_entry.hits == 1


def test_generic_client_block_restore_respects_boundary_true_off_switch(
    monkeypatch,
):
    """With MTPLX_SESSION_BOUNDARY_TRUE_RESTORE=0 the pre-v2 caution comes
    back for non-OpenCode clients: tiny-gap only, exact restore wins."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cached_tokens == exact_entry.prefix_len
    assert bank.restore_calls == 1
    assert bank.near_allow_block[0] is False
    assert block_entry.hits == 0


def test_generic_client_block_restore_respects_block_prefix_kill_switch(
    monkeypatch,
):
    """MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0 must still disable the block
    lane for generic clients even with boundary-true restore on."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cached_tokens == exact_entry.prefix_len
    assert bank.restore_calls == 1
    assert bank.near_allow_block[0] is False
    assert block_entry.hits == 0


def test_ssd_near_prefix_restore_time_is_cache_time_not_decode_time(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    ssd_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-ssd",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        cache_source="ssd",
        ssd_cache_hit=True,
        ssd_restore_s=1.25,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **_kwargs):
            return [(ssd_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert int(prefix_len) == 7
            assert mode == "clone"
            assert cache_factory is None or callable(cache_factory)
            return [], [], "clone"

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=Bank(),
    )

    assert prompt_state.cache_source == "ssd"
    assert prompt_state.ssd_cache_hit is True
    assert prompt_state.ssd_restore_s == 1.25
    assert prompt_state.cache_restore_time_s >= 1.25


def test_block_prefix_restore_matches_target_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def __init__(self):
            self.near_kwargs: list[dict[str, object]] = []

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_kwargs.append(kwargs)
            return []

    bank = Bank()
    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
    )

    assert prompt_state.cache_hit is False
    assert prompt_state.cached_tokens == 0
    assert bank.near_kwargs
    assert bank.near_kwargs[-1]["allow_block_prefix"] is True


def test_sustained_prefill_chunk_cache_cleanup_is_explicit(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP", "1")
    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        "mtplx.generation.mx.clear_cache", lambda: calls.append("clear")
    )
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert calls == ["sync", "clear", "sync", "clear"]
    assert rt.diagnostic_counters["prefill_chunk_cache_cleanup_events"] == 2


def test_sustained_prefill_stock_cache_only_requires_unsafe_allow(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_stock_cache_only_is_explicit_unsafe(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_stock_cache_only_calls"] == 2


def test_sustained_prefill_omlx_external_is_safe_profile_path(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_OMLX_EXTERNAL", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_omlx_external_calls"] == 2
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_legacy_external_prefill_routes_streamed_experts_as_prefill(monkeypatch):
    monkeypatch.setenv("MTPLX_PREFILL_OMLX_EXTERNAL", "1")

    class PhaseRecordingModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.phases: list[RoutingPhase] = []

        def __call__(self, input_ids, *, cache=None, **kwargs):
            # token_count=1 would classify as decode without an explicit
            # routing context, so a PREFILL observation proves the wrap.
            self.phases.append(current_expert_routing_phase(token_count=1))
            kwargs.pop("input_embeddings", None)
            return super().__call__(input_ids, cache=cache, **kwargs)

    model = PhaseRecordingModel()
    rt = MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=SimpleNamespace(),
    )

    assert _prefill_cache_only_forward(rt, mx.array([[7]]), cache=[]) is None
    assert (
        _prefill_cache_only_forward(
            rt,
            mx.array([[7]]),
            cache=[],
            input_embeddings=object(),
        )
        is None
    )

    assert rt.diagnostic_counters["prefill_omlx_external_calls"] == 2
    assert model.phases == [RoutingPhase.PREFILL, RoutingPhase.PREFILL]


def test_sustained_prefill_forwards_logits_controls_through_patched_kwargs_wrapper(
    monkeypatch,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = KwargsOnlyTinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0


def test_last_window_mtp_history_skips_discarded_chunk_hidden(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[tuple[list[int], int | None]] = []

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        appended.append((list(token_ids), position_offset))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    _prefill_committed_mtp_history_streaming(
        rt,
        list(range(9)),
        mtp_hidden_variant="post_norm",
        history_window_tokens=3,
    )

    assert [call["tokens"] for call in model.calls] == [2, 2, 2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert appended == [([6], 5), ([7, 8], 6)]


def test_32k_prefill_peak_memory_bounded():
    """
    Regression guard for the Ivan/Benchand 32K memory balloon.
    Run only on the Apple Silicon long-context QA machine.
    """
    if os.environ.get("MTPLX_RUN_32K_MEMORY_QA") != "1":
        pytest.skip("set MTPLX_RUN_32K_MEMORY_QA=1 on the long-context QA Mac")
    model_path = os.environ.get("MTPLX_32K_QA_MODEL")
    if not model_path:
        pytest.skip("set MTPLX_32K_QA_MODEL to a local runnable MTPLX model")

    from mtplx.runtime import load

    rt = load(model_path, mtp=True)
    text = "def f(x): return x + 1\n" * 4096
    prompt_ids = rt.tokenizer.encode(text)[:32768]
    if len(prompt_ids) < 32000:
        pytest.skip("QA prompt did not tokenize to 32K tokens")

    mx.reset_peak_memory()
    os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2048"
    os.environ["MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"] = "0"
    _prefill(rt, prompt_ids, return_hidden=True)
    peak_gb = mx.get_peak_memory() / (1024**3)

    assert peak_gb < 35.0, f"32K Sustained prefill peak was {peak_gb:.1f} GB"
