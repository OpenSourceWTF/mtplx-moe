from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


# Captured by running this test's deterministic harness against the unchanged
# implementation at 5cc2b10.  The explicit trace makes semantic diffs readable;
# stats_sha256 covers every normalized GenerationStats field, including events.
UNCHANGED_CONTROL_GOLDEN_SHA256 = (
    "3ced1d1f04e178f25d1dc3d9b92128a4d5745883660fb0214fdc9828365f8d72"
)
UNCHANGED_CONTROL_STATS_SHA256 = {
    "accept_bonus": "4771d16436768e5567310996edbabbf162260ab171b05868e47be74cf97eba90",
    "draft_stop": "8220eb6d58cabd17ff0251f735a70a3bebabf0d414eaff0e1f10683bf33785b1",
    "constraint_stop": "5e582104b1e58311cb5ea9489f376f41c6c548cbea4e01d3b777a1533bcce391",
}
UNCHANGED_CONTROL_SUMMARY = {
    "accept_bonus": {
        "tokens": [3, 4, 5, 0],
        "finish_reason": "length",
        "target_inputs": [[0, 1], [2], [3, 4, 5], [0]],
        "target_argmax": [None, [3], [4, 5, 0], [1]],
        "target_hidden_last": [[1.0, 11.0], [2.0, 12.0], [5.0, 15.0], [0.0, 10.0]],
        "target_cache_offsets": [7],
        "mtp_cache_offsets": [6],
        "commit_prefixes": [],
        "token_callbacks": [[3], [4, 5], [0]],
        "prefill_phases": ["started", "chunk", "completed"],
        "accounting": {
            "generated": 4,
            "drafted": 2,
            "accepted": 2,
            "rejected": 0,
            "correction": 0,
            "bonus": 1,
            "verify_calls": 1,
        },
        "constraint": None,
    },
    "draft_stop": {
        "tokens": [3, 4, 5],
        "finish_reason": "stop",
        "target_inputs": [[0, 1], [2], [3, 4, 5]],
        "target_argmax": [None, [3], [4, 5, 0]],
        "target_hidden_last": [[1.0, 11.0], [2.0, 12.0], [5.0, 15.0]],
        "target_cache_offsets": [6],
        "mtp_cache_offsets": [5],
        "commit_prefixes": [],
        "token_callbacks": [[3], [4]],
        "prefill_phases": ["started", "chunk", "completed"],
        "accounting": {
            "generated": 3,
            "drafted": 2,
            "accepted": 2,
            "rejected": 0,
            "correction": 0,
            "bonus": 0,
            "verify_calls": 1,
        },
        "constraint": None,
    },
    "constraint_stop": {
        "tokens": [4, 0],
        "finish_reason": "stop",
        "target_inputs": [[0, 1], [2], [4, 5, 0], [0, 1, 2]],
        "target_argmax": [None, [3], [5, 0, 1], [1, 2, 3]],
        "target_hidden_last": [[1.0, 11.0], [2.0, 12.0], [0.0, 10.0], [2.0, 12.0]],
        "target_cache_offsets": [5],
        "mtp_cache_offsets": [4],
        "commit_prefixes": [
            {"keep_tokens": 1, "verified_tokens": 3},
            {"keep_tokens": 1, "verified_tokens": 3},
        ],
        "token_callbacks": [[4], [0]],
        "prefill_phases": ["started", "chunk", "completed"],
        "accounting": {
            "generated": 2,
            "drafted": 4,
            "accepted": 0,
            "rejected": 2,
            "correction": 0,
            "bonus": 0,
            "verify_calls": 2,
        },
        "constraint": {
            "advanced": [4, 0],
            "stopped": True,
            "completed": True,
            "masked_steps": 2,
        },
    },
}


class _TraceCache:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.offset = 0

    @property
    def state(self):
        return None

    @state.setter
    def state(self, _value) -> None:
        pass

    @property
    def meta_state(self):
        return (self.offset,)

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset = int(value[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        trimmed = min(self.offset, int(count))
        self.offset -= trimmed
        return trimmed


class _Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def decode(self, tokens, **_kwargs):
        return ",".join(str(int(token)) for token in tokens)


class _GoldenModel:
    vocab = 6

    def __init__(self) -> None:
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
        self.target_trace: list[dict[str, object]] = []
        self.mtp_trace: list[dict[str, object]] = []
        self.mtp_update_trace: list[dict[str, object]] = []

    def make_cache(self):
        return [_TraceCache("target")]

    def make_mtp_cache(self):
        return [_TraceCache("mtp")]

    @staticmethod
    def _tokens(value) -> list[int]:
        return [int(token) for token in np.asarray(value).reshape(-1)]

    def _logits(self, tokens: list[int]):
        rows = []
        for token in tokens:
            row = [0.0] * self.vocab
            row[(token + 1) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    @staticmethod
    def _hidden(tokens: list[int]):
        return mx.array(
            [[[float(token), float(token + 10)] for token in tokens]],
            dtype=mx.float32,
        )

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
        tokens = self._tokens(input_ids)
        before = [int(entry.offset) for entry in cache or []]
        for entry in cache or []:
            entry.offset += len(tokens)
        keep = (
            len(tokens)
            if logits_keep is None
            else min(len(tokens), max(1, int(logits_keep)))
        )
        logits_tokens = tokens[-keep:]
        hidden = self._hidden(tokens)
        self.target_trace.append(
            {
                "input": tokens,
                "cache_before": before,
                "cache_after": [int(entry.offset) for entry in cache or []],
                "return_hidden": bool(return_hidden),
                "hidden_variant": hidden_variant,
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
                "logits_argmax": [
                    (token + 1) % self.vocab for token in logits_tokens
                ]
                if emit_logits
                else None,
                "hidden": [
                    [float(token), float(token + 10)] for token in tokens
                ],
            }
        )
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        logits = self._logits(logits_tokens)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant=None,
        position_offset=None,
    ):
        del hidden_states
        tokens = self._tokens(next_token_ids)
        before = [int(entry.offset) for entry in mtp_cache or []]
        for entry in mtp_cache or []:
            entry.offset += len(tokens)
        logits = self._logits(tokens)
        hidden = self._hidden(tokens)
        self.mtp_trace.append(
            {
                "input": tokens,
                "cache_before": before,
                "cache_after": [int(entry.offset) for entry in mtp_cache or []],
                "concat_order": concat_order,
                "return_hidden": bool(return_hidden),
                "hidden_variant": mtp_hidden_variant,
                "position_offset": position_offset,
                "logits_argmax": [
                    (token + 1) % self.vocab for token in tokens
                ],
                "hidden": [
                    [float(token), float(token + 10)] for token in tokens
                ],
            }
        )
        return (logits, hidden) if return_hidden else logits

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        mtp_hidden_variant=None,
        position_offset=None,
        **_kwargs,
    ):
        del hidden_states
        tokens = self._tokens(next_token_ids)
        before = [int(entry.offset) for entry in mtp_cache or []]
        for entry in mtp_cache or []:
            entry.offset += len(tokens)
        self.mtp_update_trace.append(
            {
                "input": tokens,
                "cache_before": before,
                "cache_after": [int(entry.offset) for entry in mtp_cache or []],
                "concat_order": concat_order,
                "hidden_variant": mtp_hidden_variant,
                "position_offset": position_offset,
            }
        )
        return None


class _ScriptConstraint:
    def __init__(self, script: list[int], vocab: int) -> None:
        self.script = list(script)
        self.vocab = int(vocab)
        self.advanced: list[int] = []
        self.masked_steps = 0
        self.mask_time_s = 0.0

    @property
    def stopped(self) -> bool:
        return len(self.advanced) >= len(self.script)

    @property
    def completed(self) -> bool:
        return self.stopped

    def mask_logits_row(self, _row):
        self.masked_steps += 1
        token = self.script[len(self.advanced)]
        values = [-1.0e9] * self.vocab
        values[token] = 10.0
        return mx.array(values, dtype=mx.float32)

    def validate_prefix(self, tokens) -> int:
        expected = self.script[len(self.advanced) :]
        matched = 0
        for actual, wanted in zip(tokens, expected):
            if int(actual) != int(wanted):
                break
            matched += 1
        return matched

    def advance_many(self, tokens) -> None:
        for token in tokens:
            expected = self.script[len(self.advanced)]
            assert int(token) == int(expected)
            self.advanced.append(int(token))


def _normalized(value, *, key: str = ""):
    if key == "peak_memory_bytes":
        return "<memory>"
    if (
        key == "timing_s"
        or key.endswith("_time_s")
        or key.endswith("_elapsed_s")
        or key.endswith("_started_s")
        or key.endswith("_tok_s")
        or key.endswith("_tps")
        or key in {"elapsed_s", "tok_s", "decode_tok_s", "end_to_end_tok_s"}
    ):
        return "<timing>"
    if isinstance(value, dict):
        return {
            str(item_key): _normalized(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _callback_event(event: dict[str, object]) -> dict[str, object]:
    stable = (
        "phase",
        "tokens_done",
        "tokens_total",
        "cached_tokens",
        "new_prefill_tokens",
    )
    return {key: event.get(key) for key in stable if key in event}


def _run_scenario(
    monkeypatch,
    *,
    name: str,
    max_tokens: int,
    stop_token_ids: set[int],
    constraint: _ScriptConstraint | None = None,
) -> dict[str, object]:
    model = _GoldenModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path(f"golden-{name}"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    token_callbacks: list[list[int]] = []
    prefill_callbacks: list[dict[str, object]] = []
    commit_prefixes: list[dict[str, int]] = []

    def commit_prefix(
        cache,
        _captures,
        *,
        keep_tokens,
        verified_tokens,
        **_kwargs,
    ):
        commit_prefixes.append(
            {
                "keep_tokens": int(keep_tokens),
                "verified_tokens": int(verified_tokens),
            }
        )
        trim = int(verified_tokens) - int(keep_tokens)
        for entry in cache:
            entry.trim(trim)
        return True

    monkeypatch.setattr("mtplx.gdn_capture.commit_captured_prefix", commit_prefix)
    output = generate_mtpk(
        runtime,
        [0, 1, 2],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=6),
        speculative_depth=2,
        seed=19,
        stop_token_ids=stop_token_ids,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        verify_core="stock",
        token_callback=lambda batch: token_callbacks.append(list(batch)),
        prefill_callback=prefill_callbacks.append,
        capture_final_state=True,
        constraint=constraint,
    )
    stats = _normalized(asdict(output.stats))
    stats_json = json.dumps(stats, sort_keys=True, separators=(",", ":"))
    final_state = output.final_state
    assert final_state is not None
    return {
        "tokens": list(output.tokens),
        "text": output.text,
        "finish_reason": output.finish_reason,
        "target_trace": model.target_trace,
        "mtp_trace": model.mtp_trace,
        "mtp_update_trace": model.mtp_update_trace,
        "target_cache_offsets": [
            int(entry.offset) for entry in final_state.final_trunk_cache
        ],
        "mtp_cache_offsets": [
            int(entry.offset)
            for entry in final_state.final_committed_mtp_cache
        ],
        "final_tokens": list(final_state.generated_token_ids),
        "final_safe_to_commit": bool(final_state.safe_to_commit),
        "final_finish_reason": final_state.finish_reason,
        "final_logits_argmax": int(mx.argmax(final_state.final_logits).item()),
        "final_hidden": np.asarray(final_state.final_hidden).tolist(),
        "commit_prefixes": commit_prefixes,
        "token_callbacks": token_callbacks,
        "prefill_callbacks": [
            _callback_event(event) for event in prefill_callbacks
        ],
        "constraint": (
            {
                "advanced": list(constraint.advanced),
                "stopped": bool(constraint.stopped),
                "completed": bool(constraint.completed),
                "masked_steps": int(constraint.masked_steps),
            }
            if constraint is not None
            else None
        ),
        "accounting": {
            "generated": int(output.stats.generated_tokens),
            "drafted": int(output.stats.drafted_tokens),
            "accepted": int(output.stats.accepted_drafts),
            "rejected": int(output.stats.rejected_drafts),
            "correction": int(output.stats.correction_tokens),
            "bonus": int(output.stats.bonus_tokens),
            "verify_calls": int(output.stats.verify_calls),
            "events": _normalized(output.stats.events),
        },
        "stats_fields": list(asdict(output.stats)),
        "stats_sha256": hashlib.sha256(stats_json.encode()).hexdigest(),
    }


def _capture_golden(monkeypatch) -> dict[str, object]:
    for key, value in {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_STATE_REBASE_EVERY": "0",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY": "off",
        "MTPLX_DROP_EVENTS": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SKIP_VERIFY_SNAPSHOT": "0",
        "MTPLX_LAZY_BONUS_VERIFY": "0",
        "MTPLX_OMIT_SPECULATIVE_BONUS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    return {
        "accept_bonus": _run_scenario(
            monkeypatch,
            name="accept-bonus",
            max_tokens=4,
            stop_token_ids=set(),
        ),
        "draft_stop": _run_scenario(
            monkeypatch,
            name="draft-stop",
            max_tokens=6,
            stop_token_ids={5},
        ),
        "constraint_stop": _run_scenario(
            monkeypatch,
            name="constraint-stop",
            max_tokens=6,
            stop_token_ids=set(),
            constraint=_ScriptConstraint([4, 0], _GoldenModel.vocab),
        ),
    }


def _semantic_summary(actual: dict[str, object]) -> dict[str, object]:
    summary = {}
    for name, raw_scenario in actual.items():
        scenario = raw_scenario
        target_trace = scenario["target_trace"]
        accounting = scenario["accounting"]
        summary[name] = {
            "tokens": scenario["tokens"],
            "finish_reason": scenario["finish_reason"],
            "target_inputs": [call["input"] for call in target_trace],
            "target_argmax": [call["logits_argmax"] for call in target_trace],
            "target_hidden_last": [
                call["hidden"][-1] for call in target_trace
            ],
            "target_cache_offsets": scenario["target_cache_offsets"],
            "mtp_cache_offsets": scenario["mtp_cache_offsets"],
            "commit_prefixes": scenario["commit_prefixes"],
            "token_callbacks": scenario["token_callbacks"],
            "prefill_phases": [
                event["phase"] for event in scenario["prefill_callbacks"]
            ],
            "accounting": {
                key: accounting[key]
                for key in (
                    "generated",
                    "drafted",
                    "accepted",
                    "rejected",
                    "correction",
                    "bonus",
                    "verify_calls",
                )
            },
            "constraint": scenario["constraint"],
        }
    return summary


def test_mtpk_request_machine_matches_unchanged_control_golden(
    monkeypatch,
) -> None:
    actual = _capture_golden(monkeypatch)
    print_mode = os.environ.get("MTPLX_PRINT_K2_GOLDEN")
    if print_mode == "sha":
        payload = json.dumps(actual, sort_keys=True, separators=(",", ":"))
        print(hashlib.sha256(payload.encode()).hexdigest())
        return
    if print_mode == "1":
        print(json.dumps(actual, sort_keys=True, separators=(",", ":")))
        return
    assert _semantic_summary(actual) == UNCHANGED_CONTROL_SUMMARY
    assert {
        name: scenario["stats_sha256"]
        for name, scenario in actual.items()
    } == UNCHANGED_CONTROL_STATS_SHA256
    payload = json.dumps(actual, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        UNCHANGED_CONTROL_GOLDEN_SHA256
    )
