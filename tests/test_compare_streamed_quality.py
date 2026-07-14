"""Deterministic quality-gate tests without loading MLX or model artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "compare_streamed_quality.py"
_QUALITY_PROMPTS = _ROOT / "benchmarks/fixtures/glm52-q2-quality-prompts.jsonl"
_BENCHMARK_PROMPT = _ROOT / "benchmarks/fixtures/glm52-q2-benchmark-prompt.txt"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_streamed_quality", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quality = _load_module()


class FakeTokenizer:
    def __init__(self, mappings: dict[str, list[int]] | None = None) -> None:
        self.mappings = mappings or {}
        self.eos_token_ids: set[int] = set()

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if text in self.mappings:
            return list(self.mappings[text])
        return [byte % 5 for byte in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)


class FakeRuntime:
    def __init__(
        self,
        *,
        tokenizer: FakeTokenizer,
        logits_by_token: dict[int, list[float]] | None = None,
        logit_scale: float = 1.0,
        events: list[str] | None = None,
        label: str = "runtime",
    ) -> None:
        self.tokenizer = tokenizer
        self.logits_by_token = logits_by_token
        self.logit_scale = logit_scale
        self.events = events if events is not None else []
        self.label = label
        self.forward_calls: list[tuple[int, tuple[int, ...]]] = []
        self.closed = False
        self.expert_streaming = SimpleNamespace(reset=lambda: None)

    def make_cache(self) -> dict[str, object]:
        return {}

    def quality_input_array(self, token_ids: list[int]) -> np.ndarray:
        return np.asarray([token_ids], dtype=np.int32)

    def admit_kv_tokens(self, _tokens: int):
        return nullcontext()

    def forward_ar(self, input_ids, *, cache=None):
        values = np.asarray(input_ids, dtype=np.int32).reshape(-1).tolist()
        self.forward_calls.append((id(cache), tuple(values)))
        rows = []
        for token in values:
            if self.logits_by_token is not None:
                row = self.logits_by_token[int(token)]
            else:
                row = [0.0] * 5
                row[(int(token) + 1) % 5] = 2.0 * self.logit_scale
            rows.append(row)
        return np.asarray([rows], dtype=np.float32)

    def close(self, *, timeout: float | None = None) -> None:
        assert timeout == 10.0
        self.closed = True
        self.events.append(f"close-{self.label}")


def _float32_nll(row: list[float], target: int) -> float:
    values = np.asarray(row, dtype=np.float32)
    maximum = np.max(values)
    logsumexp = np.float32(
        maximum + np.log(np.sum(np.exp(values - maximum), dtype=np.float32))
    )
    return float(np.float32(logsumexp - values[target]))


def test_teacher_forced_loss_aligns_next_tokens_and_reuses_chunk_cache() -> None:
    table = {
        0: [0.0, 2.0, -1.0],
        1: [1.0, 0.0, 3.0],
        2: [2.0, 1.0, 0.0],
    }
    runtime = FakeRuntime(tokenizer=FakeTokenizer(), logits_by_token=table)

    result = quality.teacher_forced_loss(runtime, [0, 1, 2, 0], chunk_tokens=2)

    expected_nll = sum(
        (
            _float32_nll(table[0], 1),
            _float32_nll(table[1], 2),
            _float32_nll(table[2], 0),
        )
    )
    assert result["token_count"] == 3
    assert result["nll_sum"] == pytest.approx(expected_nll, abs=1e-7)
    assert result["mean_nll"] == pytest.approx(expected_nll / 3, abs=1e-7)
    assert result["perplexity"] == pytest.approx(math.exp(expected_nll / 3))
    assert result["finite"] is True
    assert result["nan_count"] == 0
    assert [call[1] for call in runtime.forward_calls] == [(0, 1), (2,)]
    assert len({call[0] for call in runtime.forward_calls}) == 1


def test_greedy_diagnostics_report_agreement_and_first_divergence() -> None:
    tokenizer = FakeTokenizer({"seed": [0]})
    q4_runtime = FakeRuntime(tokenizer=tokenizer)
    q2_table = {
        0: [0.0, 2.0, 0.0, 0.0, 0.0],
        1: [0.0, 0.0, 1.0, 2.0, 0.0],
        3: [0.0, 0.0, 0.0, 0.0, 2.0],
    }
    q2_runtime = FakeRuntime(tokenizer=tokenizer, logits_by_token=q2_table)
    prompts = [{"name": "case", "category": "coding", "prompt": "seed"}]

    q4 = quality.greedy_outputs(q4_runtime, prompts, max_tokens=3)
    q2 = quality.greedy_outputs(q2_runtime, prompts, max_tokens=3)
    diagnostics = quality.greedy_diagnostics(q4, q2)

    assert q4[0]["token_ids"] == [1, 2, 3]
    assert q2[0]["token_ids"] == [1, 3, 4]
    assert diagnostics["agreement_tokens"] == 1
    assert diagnostics["compared_positions"] == 3
    assert diagnostics["agreement_fraction"] == pytest.approx(1 / 3)
    assert diagnostics["first_divergence"] == {
        "prompt_index": 0,
        "prompt_name": "case",
        "token_index": 1,
        "q4_token": 2,
        "q2_token": 3,
    }


def _write_lane_files(root: Path, manifest_hash: str) -> tuple[Path, Path]:
    root.mkdir()
    (root / "tokenizer.json").write_text('{"fixture":"same"}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        '{"add_bos_token":false}\n', encoding="utf-8"
    )
    manifest = root / "expert-manifest.json"
    manifest.write_text(
        json.dumps({"manifest_sha256": manifest_hash}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, manifest


def _combined_corpus_hash(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def test_compare_quality_records_receipts_and_never_co_resides_lanes(
    tmp_path: Path,
) -> None:
    q4_root, q4_manifest = _write_lane_files(tmp_path / "q4", "a" * 64)
    q2_root, q2_manifest = _write_lane_files(tmp_path / "q2", "b" * 64)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"abc")
    second.write_bytes(b"de")
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        json.dumps({"name": "p", "category": "coding", "prompt": "seed"}) + "\n",
        encoding="utf-8",
    )
    events: list[str] = []
    tokenizer = FakeTokenizer({"abc": [0, 1, 2], "de": [3, 4], "seed": [0]})
    q4_runtime = FakeRuntime(tokenizer=tokenizer, events=events, label="q4")
    q2_runtime = FakeRuntime(tokenizer=tokenizer, events=events, label="q2")

    def load_q4():
        events.append("load-q4")
        return q4_runtime

    def clear_q4() -> None:
        assert q4_runtime.closed is True
        events.append("clear-q4")

    def load_q2():
        assert events[-2:] == ["close-q4", "clear-q4"]
        events.append("load-q2")
        return q2_runtime

    def clear_q2() -> None:
        assert q2_runtime.closed is True
        events.append("clear-q2")

    q4_lane = quality.QualityLane(
        config=quality.LaneConfig("q4", q4_root, q4_manifest, "glm52-q4"),
        load_runtime=load_q4,
        clear_cache=clear_q4,
    )
    q2_lane = quality.QualityLane(
        config=quality.LaneConfig("q2", q2_root, q2_manifest, "glm52-expert-q2"),
        load_runtime=load_q2,
        clear_cache=clear_q2,
    )

    result = quality.compare_quality(
        q4_lane,
        q2_lane,
        corpus_files=[first, second],
        prompt_file=prompt_file,
        evaluation_tokens=5,
        chunk_tokens=2,
        greedy_max_tokens=3,
        max_relative_perplexity_regression=0.05,
    )

    assert events == [
        "load-q4",
        "close-q4",
        "clear-q4",
        "load-q2",
        "close-q2",
        "clear-q2",
    ]
    assert result["corpus"]["file_order"] == [str(first), str(second)]
    assert result["corpus"]["sha256"] == _combined_corpus_hash([b"abc", b"de"])
    assert result["corpus"]["token_count"] == 5
    assert result["lanes"]["q4"]["loss"]["token_count"] == 4
    assert result["lanes"]["q4"]["manifest"]["declared_sha256"] == "a" * 64
    assert result["lanes"]["q2"]["manifest"]["declared_sha256"] == "b" * 64
    assert len(result["lanes"]["q4"]["manifest"]["file_sha256"]) == 64
    assert (
        result["lanes"]["q4"]["tokenizer"]["sha256"]
        == result["lanes"]["q2"]["tokenizer"]["sha256"]
    )
    assert result["relative_perplexity_regression"] == pytest.approx(0.0)
    assert result["quality_passed"] is True
    assert result["passed"] is True
    assert result["errors"] == []


def test_q4_load_failure_clears_cache_and_does_not_start_q2(tmp_path: Path) -> None:
    q4_root, q4_manifest = _write_lane_files(tmp_path / "failed-q4", "a" * 64)
    q2_root, q2_manifest = _write_lane_files(tmp_path / "unused-q2", "b" * 64)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc", encoding="utf-8")
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"name":"p","category":"coding","prompt":"seed"}\n',
        encoding="utf-8",
    )
    events: list[str] = []

    def load_q4():
        events.append("load-q4")
        raise RuntimeError("partial Q4 load failed")

    def load_q2():
        events.append("load-q2")
        raise AssertionError("Q2 must not start after a Q4 operational failure")

    q4_lane = quality.QualityLane(
        config=quality.LaneConfig("q4", q4_root, q4_manifest, "glm52-q4"),
        load_runtime=load_q4,
        clear_cache=lambda: events.append("clear-q4"),
    )
    q2_lane = quality.QualityLane(
        config=quality.LaneConfig("q2", q2_root, q2_manifest, "glm52-expert-q2"),
        load_runtime=load_q2,
        clear_cache=lambda: events.append("clear-q2"),
    )

    result = quality.compare_quality(
        q4_lane,
        q2_lane,
        corpus_files=[corpus],
        prompt_file=prompt_file,
        evaluation_tokens=3,
        chunk_tokens=2,
        greedy_max_tokens=2,
    )

    assert events == ["load-q4", "clear-q4"]
    assert result["passed"] is False
    assert result["lanes"]["q2"]["skipped"] == "q4 lane did not complete safely"
    assert result["errors"][0]["stage"] == "lane_evaluation"


@pytest.mark.parametrize(
    ("q4_perplexity", "q2_perplexity", "finite", "regression", "passed"),
    (
        (10.0, 10.4, True, 0.04, True),
        (10.0, 10.6, True, 0.06, False),
        (10.0, 10.0, False, 0.0, False),
    ),
)
def test_quality_gate_is_finite_and_at_most_five_percent(
    q4_perplexity: float,
    q2_perplexity: float,
    finite: bool,
    regression: float,
    passed: bool,
) -> None:
    result = quality.quality_gate(
        q4_perplexity,
        q2_perplexity,
        finite=finite,
        max_relative_perplexity_regression=0.05,
    )

    assert result["relative_perplexity_regression"] == pytest.approx(regression)
    assert result["quality_passed"] is passed


def _cli_args(tmp_path: Path, output: Path) -> list[str]:
    q4_root, q4_manifest = _write_lane_files(tmp_path / "cli-q4", "c" * 64)
    q2_root, q2_manifest = _write_lane_files(tmp_path / "cli-q2", "d" * 64)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("corpus", encoding="utf-8")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"name":"p","category":"coding","prompt":"seed"}\n',
        encoding="utf-8",
    )
    return [
        "--q4-root",
        str(q4_root),
        "--q4-manifest",
        str(q4_manifest),
        "--q4-model-key",
        "glm52-q4",
        "--q2-root",
        str(q2_root),
        "--q2-manifest",
        str(q2_manifest),
        "--q2-model-key",
        "glm52-expert-q2",
        "--memory-limit",
        "160GiB",
        "--expert-cache-limit",
        "96GiB",
        "--runtime-reserve",
        "16GiB",
        "--max-live-kv-tokens",
        "8192",
        "--corpus-file",
        str(corpus),
        "--evaluation-tokens",
        "64",
        "--chunk-tokens",
        "8",
        "--prompt-file",
        str(prompts),
        "--greedy-max-tokens",
        "4",
        "--max-relative-perplexity-regression",
        "0.05",
        "--output-json",
        str(output),
    ]


def test_cli_writes_complete_json_before_quality_exit_two(tmp_path: Path) -> None:
    output = tmp_path / "rejected.json"

    def rejected(*_args, **_kwargs):
        return {
            "schema": "mtplx-streamed-quality-v1",
            "passed": False,
            "quality_passed": False,
            "relative_perplexity_regression": 0.06,
            "errors": [],
        }

    exit_code = quality.main(_cli_args(tmp_path, output), _compare_quality=rejected)

    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "mtplx-streamed-quality-v1"
    assert payload["passed"] is False
    assert payload["relative_perplexity_regression"] == pytest.approx(0.06)


def test_cli_operational_failure_returns_one_and_writes_error_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "operational-error.json"

    def failed(*_args, **_kwargs):
        raise RuntimeError("loader failed")

    exit_code = quality.main(_cli_args(tmp_path, output), _compare_quality=failed)

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["quality_passed"] is False
    assert payload["errors"] == [
        {"type": "RuntimeError", "message": "loader failed", "stage": "operation"}
    ]


def test_reviewed_prompt_fixtures_cover_required_categories() -> None:
    prompts = [
        json.loads(line)
        for line in _QUALITY_PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {item["category"] for item in prompts} >= {
        "coding",
        "mathematical_reasoning",
        "structured_extraction",
        "long_form_explanation",
    }
    assert all(item["name"] and item["prompt"].strip() for item in prompts)
    benchmark_prompt = _BENCHMARK_PROMPT.read_bytes()
    assert benchmark_prompt.endswith(b"\n")
    assert b"repository" in benchmark_prompt.lower()
    assert b"tests" in benchmark_prompt.lower()
