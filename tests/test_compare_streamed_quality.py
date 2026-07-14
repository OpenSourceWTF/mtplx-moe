"""Deterministic quality-gate tests without loading MLX or model artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "compare_streamed_quality.py"
_QUALITY_PROMPTS = _ROOT / "benchmarks/fixtures/glm52-q2-quality-prompts.jsonl"
_BENCHMARK_PROMPT = _ROOT / "benchmarks/fixtures/glm52-q2-benchmark-prompt.txt"
_HY3_QUALITY_PROMPTS = _ROOT / "benchmarks/fixtures/hy3-q2-quality-prompts.jsonl"
_HY3_BENCHMARK_PROMPT = _ROOT / "benchmarks/fixtures/hy3-q2-benchmark-prompt.txt"


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


def _write_hf_snapshot_member(
    tmp_path: Path,
    *,
    revision: str = "a" * 40,
    member_name: str = "tokenizer.json",
    payload: bytes = b'{"fixture":"hf-snapshot"}\n',
    blob_name: str | None = None,
    target: str | None = None,
) -> tuple[Path, Path, Path]:
    repository = tmp_path / "models--example--glm"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    selected_blob_name = blob_name or hashlib.sha256(payload).hexdigest()
    blob = blobs / selected_blob_name
    blob.write_bytes(payload)
    member = snapshot / member_name
    member.symlink_to(target or f"../../blobs/{selected_blob_name}")
    return repository, snapshot, blob


def _git_blob_sha1(payload: bytes) -> str:
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def test_tokenizer_receipt_accepts_identity_bound_hf_snapshot_symlink(
    tmp_path: Path,
) -> None:
    _repository, snapshot, blob = _write_hf_snapshot_member(tmp_path)

    receipt = quality._tokenizer_receipt(snapshot)

    assert receipt["files"] == [
        {
            "name": "tokenizer.json",
            "path": str(snapshot / "tokenizer.json"),
            "bytes": blob.stat().st_size,
            "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        }
    ]


def test_hf_snapshot_accepts_git_blob_sha1_content_address(tmp_path: Path) -> None:
    payload = b'{"fixture":"git-blob"}\n'
    _repository, snapshot, blob = _write_hf_snapshot_member(
        tmp_path,
        payload=payload,
        blob_name=_git_blob_sha1(payload),
    )

    receipt = quality._tokenizer_receipt(snapshot)

    assert receipt["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["files"][0]["bytes"] == blob.stat().st_size


def test_hf_snapshot_rejects_invalid_git_blob_content_address(tmp_path: Path) -> None:
    _repository, snapshot, _blob = _write_hf_snapshot_member(
        tmp_path,
        blob_name="b" * 40,
    )

    with pytest.raises(ValueError, match="content address"):
        quality._tokenizer_receipt(snapshot)


def test_hf_snapshot_rejects_blobs_swap_before_first_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, snapshot, blob = _write_hf_snapshot_member(tmp_path)
    replacement = repository / "replacement-blobs"
    replacement.mkdir()
    (replacement / blob.name).write_bytes(b"attacker bytes under the same name\n")
    real_stat = quality.os.stat
    swapped = False

    def racing_stat(path, *args, **kwargs):
        nonlocal swapped
        if path == "blobs" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            (repository / "blobs").rename(repository / "original-blobs")
            replacement.rename(repository / "blobs")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(quality.os, "stat", racing_stat)

    with pytest.raises(ValueError, match="content address"):
        quality._tokenizer_receipt(snapshot)


def test_hf_tokenizer_receipt_rejects_revision_swap_between_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, snapshot, _blob = _write_hf_snapshot_member(tmp_path)
    config_payload = b'{"fixture":"original-config"}\n'
    config_blob_name = hashlib.sha256(config_payload).hexdigest()
    (repository / "blobs" / config_blob_name).write_bytes(config_payload)
    (snapshot / "tokenizer_config.json").symlink_to(f"../../blobs/{config_blob_name}")
    replacement = repository / "replacement-snapshots"
    replacement_revision = replacement / snapshot.name
    replacement_revision.mkdir(parents=True)
    attacker_payload = b'{"fixture":"replacement-config"}\n'
    attacker_blob_name = hashlib.sha256(attacker_payload).hexdigest()
    (repository / "blobs" / attacker_blob_name).write_bytes(attacker_payload)
    (replacement_revision / "tokenizer_config.json").symlink_to(
        f"../../blobs/{attacker_blob_name}"
    )
    real_read = quality.os.read
    swapped = False

    def racing_read(descriptor, size):
        nonlocal swapped
        payload = real_read(descriptor, size)
        if not swapped:
            swapped = True
            (repository / "snapshots").rename(repository / "original-snapshots")
            replacement.rename(repository / "snapshots")
        return payload

    monkeypatch.setattr(quality.os, "read", racing_read)

    with pytest.raises(ValueError, match="changed during tokenizer receipt"):
        quality._tokenizer_receipt(snapshot)


def test_regular_tokenizer_receipt_rejects_root_swap_between_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "q2-root"
    root.mkdir()
    (root / "tokenizer.json").write_text('{"fixture":"original"}\n')
    (root / "tokenizer_config.json").write_text('{"fixture":"original"}\n')
    replacement = tmp_path / "replacement-root"
    replacement.mkdir()
    (replacement / "tokenizer_config.json").write_text('{"fixture":"replacement"}\n')
    real_read = quality.os.read
    swapped = False

    def racing_read(descriptor, size):
        nonlocal swapped
        payload = real_read(descriptor, size)
        if not swapped:
            swapped = True
            root.rename(tmp_path / "original-root")
            replacement.rename(root)
        return payload

    monkeypatch.setattr(quality.os, "read", racing_read)

    with pytest.raises(ValueError, match="changed during tokenizer receipt"):
        quality._tokenizer_receipt(root)


def test_regular_tokenizer_receipt_rejects_member_swap_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "q2-root"
    root.mkdir()
    member = root / "tokenizer.json"
    member.write_text('{"fixture":"original"}\n')
    (root / "tokenizer_config.json").write_text('{"fixture":"config"}\n')
    replacement = tmp_path / "replacement-tokenizer.json"
    replacement.write_text('{"fixture":"replacement"}\n')
    real_read = quality._read_stable_descriptor
    swapped = False

    def racing_read(descriptor, *, label, path):
        nonlocal swapped
        payload = real_read(descriptor, label=label, path=path)
        if not swapped:
            swapped = True
            member.rename(root / "original-tokenizer.json")
            replacement.rename(member)
        return payload

    monkeypatch.setattr(quality, "_read_stable_descriptor", racing_read)

    with pytest.raises(ValueError, match="changed during tokenizer receipt"):
        quality._tokenizer_receipt(root)


def test_hf_snapshot_symlink_requires_a_pinned_revision(tmp_path: Path) -> None:
    _repository, snapshot, _blob = _write_hf_snapshot_member(
        tmp_path,
        revision="main",
    )

    with pytest.raises(ValueError, match="pinned revision"):
        quality._stable_file_bytes(snapshot / "tokenizer.json", label="tokenizer")


@pytest.mark.parametrize(
    "target",
    (
        "../../../outside",
        "../../blobs/../outside",
        "/tmp/outside",
        "../../blobs/nested/member",
    ),
)
def test_hf_snapshot_symlink_rejects_noncanonical_targets(
    tmp_path: Path,
    target: str,
) -> None:
    _repository, snapshot, _blob = _write_hf_snapshot_member(
        tmp_path,
        target=target,
    )

    with pytest.raises(ValueError, match="exact ../../blobs/<flat-name>"):
        quality._stable_file_bytes(snapshot / "tokenizer.json", label="tokenizer")


def test_hf_snapshot_symlink_rejects_snapshots_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, snapshot, _blob = _write_hf_snapshot_member(tmp_path)
    revision = snapshot.name
    replacement = repository / "replacement-snapshots"
    replacement_member = replacement / revision / "tokenizer.json"
    replacement_member.parent.mkdir(parents=True)
    replacement_member.symlink_to(snapshot.joinpath("tokenizer.json").readlink())
    real_open = quality.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "snapshots" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            (repository / "snapshots").rename(repository / "original-snapshots")
            replacement.rename(repository / "snapshots")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(quality.os, "open", racing_open)

    with pytest.raises(ValueError, match="changed while being opened"):
        quality._stable_file_bytes(snapshot / "tokenizer.json", label="tokenizer")


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


def test_nonfinite_loss_is_json_safe_and_remains_a_quality_rejection(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        tokenizer=FakeTokenizer(),
        logits_by_token={0: [0.0, math.nan], 1: [0.0, 1.0]},
    )

    loss = quality.teacher_forced_loss(runtime, [0, 1], chunk_tokens=1)

    assert loss["finite"] is False
    assert loss["nll_sum"] is None
    assert loss["mean_nll"] is None
    assert loss["perplexity"] is None
    assert loss["error"]["type"] == "NonFiniteQualityEvidence"
    payload = {
        "schema": "mtplx-streamed-quality-v1",
        "passed": False,
        "quality_passed": False,
        "relative_perplexity_regression": None,
        "lanes": {"q4": {"loss": loss}},
        "errors": [],
    }
    output = tmp_path / "nonfinite.json"

    exit_code = quality.main(
        _cli_args(tmp_path, output),
        _compare_quality=lambda *_args, **_kwargs: payload,
    )

    assert exit_code == 2
    raw = output.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw)["lanes"]["q4"]["loss"]["perplexity"] is None


def test_json_writer_rejects_unhandled_nonfinite_values(tmp_path: Path) -> None:
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        quality._write_json_once(output, {"unhandled": math.nan})

    assert not output.exists()


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


def _write_lane_files(
    root: Path,
    manifest_hash: str,
    *,
    model_key: str = "glm52-q4",
    resident_payload: bytes = b"resident fixture\n",
    tokenizer_payload: str = '{"fixture":"same"}\n',
    index_indent: int | None = None,
) -> tuple[Path, Path]:
    root.mkdir()
    (root / "tokenizer.json").write_text(tokenizer_payload, encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        '{"add_bos_token":false}\n', encoding="utf-8"
    )
    resident_name = "model-00001-of-00001.safetensors"
    (root / resident_name).write_bytes(resident_payload)
    index = {
        "metadata": {"total_size": len(resident_payload)},
        "weight_map": {"model.embed_tokens.weight": resident_name},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=index_indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = root / "expert-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_hash,
                "model_key": model_key,
                "shards": [
                    {
                        "name": resident_name,
                        "size": len(resident_payload),
                        "sha256": hashlib.sha256(resident_payload).hexdigest(),
                    }
                ],
                "resident_tensors": [
                    {
                        "tensor": "model.embed_tokens.weight",
                        "shard": resident_name,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
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
    q2_root, q2_manifest = _write_lane_files(
        tmp_path / "q2", "b" * 64, model_key="glm52-expert-q2"
    )
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
    assert (
        result["lanes"]["q4"]["artifact"]["index"]["sha256"]
        == result["lanes"]["q2"]["artifact"]["index"]["sha256"]
    )
    assert (
        result["lanes"]["q4"]["artifact"]["residents"]["sha256"]
        == result["lanes"]["q2"]["artifact"]["residents"]["sha256"]
    )
    expected_resident_sha256 = hashlib.sha256(b"resident fixture\n").hexdigest()
    assert (
        result["lanes"]["q4"]["artifact"]["residents"]["files"][0]["sha256"]
        == expected_resident_sha256
    )
    assert (
        result["lanes"]["q2"]["artifact"]["residents"]["files"][0]["sha256"]
        == expected_resident_sha256
    )
    assert result["relative_perplexity_regression"] == pytest.approx(0.0)
    assert result["quality_passed"] is True
    assert result["passed"] is True
    assert result["errors"] == []


def test_q4_load_failure_clears_cache_and_does_not_start_q2(tmp_path: Path) -> None:
    q4_root, q4_manifest = _write_lane_files(tmp_path / "failed-q4", "a" * 64)
    q2_root, q2_manifest = _write_lane_files(
        tmp_path / "unused-q2", "b" * 64, model_key="glm52-expert-q2"
    )
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
    ("mismatch", "expected_stage"),
    (
        ("resident", "resident_identity"),
        ("index", "resident_index_identity"),
        ("tokenizer", "tokenizer_identity"),
    ),
)
def test_hy3_rejects_nonidentical_lane_artifact_receipts(
    tmp_path: Path,
    mismatch: str,
    expected_stage: str,
) -> None:
    q4_root, q4_manifest = _write_lane_files(
        tmp_path / "hy3-q4",
        "a" * 64,
        model_key="hy3-expert-only-q4",
    )
    q2_root, q2_manifest = _write_lane_files(
        tmp_path / "hy3-q2",
        "b" * 64,
        model_key="hy3-expert-q2",
        resident_payload=(
            b"different resident fixture\n"
            if mismatch == "resident"
            else b"resident fixture\n"
        ),
        tokenizer_payload=(
            '{"fixture":"different"}\n'
            if mismatch == "tokenizer"
            else '{"fixture":"same"}\n'
        ),
        index_indent=2 if mismatch == "index" else None,
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc", encoding="utf-8")
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"name":"p","category":"coding","prompt":"seed"}\n',
        encoding="utf-8",
    )
    tokenizer = FakeTokenizer({"abc": [0, 1, 2], "seed": [0]})
    q4_runtime = FakeRuntime(tokenizer=tokenizer, label="q4")
    q2_runtime = FakeRuntime(tokenizer=tokenizer, label="q2")
    q4_lane = quality.QualityLane(
        config=quality.LaneConfig("q4", q4_root, q4_manifest, "hy3-expert-only-q4"),
        load_runtime=lambda: q4_runtime,
        clear_cache=lambda: None,
    )
    q2_lane = quality.QualityLane(
        config=quality.LaneConfig("q2", q2_root, q2_manifest, "hy3-expert-q2"),
        load_runtime=lambda: q2_runtime,
        clear_cache=lambda: None,
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

    assert result["passed"] is False
    assert any(error["stage"] == expected_stage for error in result["errors"])


def test_hy3_rejects_a_lane_key_that_differs_from_its_manifest(tmp_path: Path) -> None:
    root, manifest = _write_lane_files(
        tmp_path / "hy3-q2",
        "b" * 64,
        model_key="hy3-expert-q2",
    )

    with pytest.raises(ValueError, match="exactly match"):
        quality._artifact_receipt(
            root,
            manifest,
            expected_model_key="hy3-expert-only-q4",
        )


def test_hy3_resident_receipt_rejects_same_size_content_corruption(
    tmp_path: Path,
) -> None:
    original = b"resident fixture\n"
    root, manifest = _write_lane_files(
        tmp_path / "hy3-q2",
        "b" * 64,
        model_key="hy3-expert-q2",
        resident_payload=original,
    )
    resident = root / "model-00001-of-00001.safetensors"
    resident.write_bytes(b"X" * len(original))

    with pytest.raises(ValueError, match="resident shard hash"):
        quality._artifact_receipt(
            root,
            manifest,
            expected_model_key="hy3-expert-q2",
        )


def test_hy3_resident_receipt_rejects_member_replacement_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _write_lane_files(
        tmp_path / "hy3-q2",
        "b" * 64,
        model_key="hy3-expert-q2",
    )
    resident = root / "model-00001-of-00001.safetensors"
    resident_inode = resident.stat().st_ino
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"X" * resident.stat().st_size)
    real_read = quality.os.read
    swapped = False

    def racing_read(descriptor, size):
        nonlocal swapped
        payload = real_read(descriptor, size)
        if not swapped and quality.os.fstat(descriptor).st_ino == resident_inode:
            swapped = True
            resident.rename(root / "original.safetensors")
            replacement.rename(resident)
        return payload

    monkeypatch.setattr(quality.os, "read", racing_read)

    with pytest.raises(ValueError, match="changed while being hashed"):
        quality._artifact_receipt(
            root,
            manifest,
            expected_model_key="hy3-expert-q2",
        )

    assert swapped is True


def test_hy3_resident_receipt_rejects_root_replacement_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _write_lane_files(
        tmp_path / "hy3-q2",
        "b" * 64,
        model_key="hy3-expert-q2",
    )
    resident = root / "model-00001-of-00001.safetensors"
    resident_inode = resident.stat().st_ino
    replacement = tmp_path / "replacement-root"
    shutil.copytree(root, replacement)
    replacement_resident = replacement / resident.name
    replacement_resident.write_bytes(b"X" * replacement_resident.stat().st_size)
    original_root = tmp_path / "original-root"
    real_read = quality.os.read
    swapped = False

    def racing_read(descriptor, size):
        nonlocal swapped
        payload = real_read(descriptor, size)
        if not swapped and quality.os.fstat(descriptor).st_ino == resident_inode:
            swapped = True
            root.rename(original_root)
            replacement.rename(root)
        return payload

    monkeypatch.setattr(quality.os, "read", racing_read)

    with pytest.raises(ValueError, match="model root changed"):
        quality._artifact_receipt(
            root,
            manifest,
            expected_model_key="hy3-expert-q2",
        )

    assert swapped is True


def _install_fake_lane_modules(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
    *,
    manifest_value: object,
) -> type:
    expert_manifest = ModuleType("mtplx.expert_manifest")
    expert_runtime = ModuleType("mtplx.expert_runtime")
    runtime_module = ModuleType("mtplx.runtime")

    class FakeStreamingConfig:
        def __init__(self, **kwargs) -> None:
            captured["streaming"] = kwargs

    def fake_load_manifest(path, *, verify_digest=True):
        captured["loaded_manifest"] = (path, verify_digest)
        return manifest_value

    def fake_load(root, **kwargs):
        captured["root"] = root
        captured["load"] = kwargs
        return object()

    expert_manifest.load_expert_manifest = fake_load_manifest
    expert_runtime.ExpertStreamingConfig = FakeStreamingConfig

    def fake_parse_memory_bytes(value):
        if value == "16GiB":
            return 16 * 1024**3
        if value == "8MiB":
            return 8 * 1024**2
        return int(value)

    expert_runtime.parse_memory_bytes = fake_parse_memory_bytes
    runtime_module.load = fake_load
    monkeypatch.setitem(sys.modules, "mtplx.expert_manifest", expert_manifest)
    monkeypatch.setitem(sys.modules, "mtplx.expert_runtime", expert_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.runtime", runtime_module)
    return FakeStreamingConfig


def test_hy3_trusted_sidecar_mode_rejects_a_manifest_without_a_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_lane_modules(
        monkeypatch,
        captured,
        manifest_value=SimpleNamespace(model_key="hy3-expert-q2", sidecar=None),
    )
    root = tmp_path / "hy3"
    root.mkdir()
    manifest = root / "expert-manifest.json"
    config = quality.LaneConfig(
        "q2",
        root,
        manifest,
        "hy3-expert-q2",
        memory_limit="120259084288",
        expert_cache_limit="83034243072",
        trust_sidecar=True,
    )

    with pytest.raises(ValueError, match="validated manifest sidecar"):
        quality._load_lane_runtime(config)

    assert "streaming" not in captured
    assert "load" not in captured


def test_hy3_trusted_sidecar_mode_rejects_a_missing_sidecar_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_lane_modules(
        monkeypatch,
        captured,
        manifest_value=SimpleNamespace(
            model_key="hy3-expert-q2",
            sidecar=SimpleNamespace(file="experts.bin", size=4096),
        ),
    )
    root = tmp_path / "hy3"
    root.mkdir()
    manifest = root / "expert-manifest.json"
    config = quality.LaneConfig(
        "q2",
        root,
        manifest,
        "hy3-expert-q2",
        memory_limit="120259084288",
        expert_cache_limit="83034243072",
        trust_sidecar=True,
    )

    with pytest.raises((OSError, ValueError), match="experts.bin|sidecar"):
        quality._load_lane_runtime(config)

    assert "streaming" not in captured
    assert "load" not in captured


def test_hy3_source_segment_fallback_keeps_record_hashing_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_lane_modules(
        monkeypatch,
        captured,
        manifest_value=SimpleNamespace(model_key="hy3-expert-q2", sidecar=None),
    )
    root = tmp_path / "hy3"
    manifest = root / "expert-manifest.json"
    config = quality.LaneConfig(
        "q2",
        root,
        manifest,
        "hy3-expert-q2",
        memory_limit="120259084288",
        expert_cache_limit="83034243072",
        trust_sidecar=False,
    )

    quality._load_lane_runtime(config)

    streaming = captured["streaming"]
    assert isinstance(streaming, dict)
    assert streaming["verify_record_hashes"] is True
    assert "loaded_manifest" not in captured


def test_hy3_runtime_configuration_forces_ar_and_carries_streaming_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    root = tmp_path / "hy3"
    root.mkdir()
    sidecar_payload = b"trusted sidecar fixture\n"
    (root / "experts.bin").write_bytes(sidecar_payload)
    manifest = root / "expert-manifest.json"
    FakeStreamingConfig = _install_fake_lane_modules(
        monkeypatch,
        captured,
        manifest_value=SimpleNamespace(
            model_key="hy3-expert-q2",
            sidecar=SimpleNamespace(file="experts.bin", size=len(sidecar_payload)),
        ),
    )
    config = quality.LaneConfig(
        "q2",
        root,
        manifest,
        "hy3-expert-q2",
        memory_limit="120259084288",
        expert_cache_limit="83034243072",
        runtime_reserve="8589934592",
        max_live_kv_tokens=18888,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="component-banks",
        transient_slots=32,
        read_chunk="67108864",
        f_nocache=True,
        trust_sidecar=True,
    )

    quality._load_lane_runtime(config)

    assert captured["root"] == root
    assert captured["loaded_manifest"] == (manifest, True)
    load_kwargs = captured["load"]
    assert isinstance(load_kwargs, dict)
    assert load_kwargs["mtp"] is False
    assert load_kwargs["expert_manifest"] == manifest
    assert isinstance(load_kwargs["expert_streaming_config"], FakeStreamingConfig)
    assert captured["streaming"] == {
        "model_key": "hy3-expert-q2",
        "memory_limit_bytes": 120259084288,
        "expert_cache_limit_bytes": 83034243072,
        "runtime_reserve_bytes": 8589934592,
        "max_live_kv_tokens": 18888,
        "cache_policy": "lru",
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "transient_slots": 32,
        "max_read_chunk_bytes": 67108864,
        "bypass_page_cache": True,
        "verify_record_hashes": False,
    }


@pytest.mark.parametrize(
    ("q4_perplexity", "q2_perplexity", "finite", "regression", "passed"),
    (
        (10.0, 10.4, True, 0.04, True),
        (10.0, 10.5, True, 0.05, True),
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


def test_quality_gate_rejects_a_threshold_above_five_percent() -> None:
    with pytest.raises(ValueError, match="must not exceed 0.05"):
        quality.quality_gate(
            10.0,
            10.5,
            finite=True,
            max_relative_perplexity_regression=0.0500001,
        )


def test_quality_gate_rejects_the_next_threshold_float_above_five_percent() -> None:
    with pytest.raises(ValueError, match="must not exceed 0.05"):
        quality.quality_gate(
            10.0,
            10.5,
            finite=True,
            max_relative_perplexity_regression=math.nextafter(0.05, math.inf),
        )


def test_quality_gate_rejects_the_next_float_above_five_percent() -> None:
    q2_perplexity = math.nextafter(10.5, math.inf)

    result = quality.quality_gate(
        10.0,
        q2_perplexity,
        finite=True,
        max_relative_perplexity_regression=0.05,
    )

    assert result["relative_perplexity_regression"] > 0.05
    assert result["quality_passed"] is False


def test_quality_gate_records_nonfinite_perplexity_as_json_safe_error() -> None:
    result = quality.quality_gate(
        None,
        10.0,
        finite=False,
        max_relative_perplexity_regression=0.05,
    )

    assert result["relative_perplexity_regression"] is None
    assert result["quality_passed"] is False
    assert result["error"]["type"] == "NonFiniteQualityEvidence"


def _cli_args(tmp_path: Path, output: Path) -> list[str]:
    q4_root, q4_manifest = _write_lane_files(tmp_path / "cli-q4", "c" * 64)
    q2_root, q2_manifest = _write_lane_files(
        tmp_path / "cli-q2", "d" * 64, model_key="glm52-expert-q2"
    )
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


def test_hy3_cli_accepts_the_planned_streaming_configuration(tmp_path: Path) -> None:
    args = _cli_args(tmp_path, tmp_path / "hy3.json")
    args[args.index("--q4-model-key") + 1] = "hy3-expert-only-q4"
    args[args.index("--q2-model-key") + 1] = "hy3-expert-q2"
    output_index = args.index("--output-json")
    args[output_index:output_index] = [
        "--cache-policy",
        "lru",
        "--cache-scope",
        "global",
        "--slot-layout",
        "component-banks",
        "--transient-slots",
        "32",
        "--read-chunk",
        "67108864",
        "--f-nocache",
        "--trust-sidecar",
    ]

    parsed = quality.build_parser().parse_args(args)

    assert parsed.q4_model_key == "hy3-expert-only-q4"
    assert parsed.q2_model_key == "hy3-expert-q2"
    assert parsed.cache_policy == "lru"
    assert parsed.cache_scope == "global"
    assert parsed.slot_layout == "component-banks"
    assert parsed.transient_slots == 32
    assert parsed.read_chunk == "67108864"
    assert parsed.f_nocache is True
    assert parsed.trust_sidecar is True


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


def test_cli_rejects_a_quality_ceiling_above_five_percent(tmp_path: Path) -> None:
    args = _cli_args(tmp_path, tmp_path / "must-not-write.json")
    threshold_index = args.index("--max-relative-perplexity-regression") + 1
    args[threshold_index] = "0.0500001"

    with pytest.raises(SystemExit) as exc_info:
        quality.build_parser().parse_args(args)

    assert exc_info.value.code == 2


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


def test_hy3_reviewed_prompt_fixtures_cover_required_categories() -> None:
    prompts = [
        json.loads(line)
        for line in _HY3_QUALITY_PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {item["category"] for item in prompts} >= {
        "coding",
        "mathematical_reasoning",
        "structured_extraction",
        "long_form_explanation",
    }
    assert all(item["name"] and item["prompt"].strip() for item in prompts)
    benchmark_prompt = _HY3_BENCHMARK_PROMPT.read_bytes()
    assert benchmark_prompt.endswith(b"\n")
    assert b"repository" in benchmark_prompt.lower()
    assert b"tests" in benchmark_prompt.lower()
