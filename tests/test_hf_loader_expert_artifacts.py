"""Expert-artifact completeness and opt-out download patterns."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.hf_loader import (
    EXPERT_BANK_IGNORE_PATTERNS,
    PARTIAL_DOWNLOAD_MARKER,
    cached_model_incompleteness_reason,
    cached_model_is_complete,
    expert_artifact_status,
    partial_download_info,
    pull_model,
    repo_file_is_ignored,
    resolve_model_path,
    validate_mtplx_model_files,
)


def _resident_model(root: Path) -> Path:
    """A complete non-expert artifact: config + index + the shard it names."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return root


def _write_expert_manifest(
    root: Path, *, sidecar_file: str | None = "experts.bin", size: int = 16
) -> None:
    manifest: dict[str, object] = {
        "format": "mtplx-expert-manifest-v1",
        "model_key": "fixture-expert-q2",
        "source_repo": "local/fixture",
        "source_revision": "0" * 40,
        "quantization": {"bits": 2, "group_size": 32, "mode": "affine"},
        "artifact": {
            "tensor_bytes": size,
            "resident_tensor_bytes": 0,
            "routed_expert_bytes": size,
            "shard_count": 0,
            "record_count": 0,
        },
        "shards": [],
        "resident_tensors": [],
        "records": [],
    }
    if sidecar_file is not None:
        manifest["sidecar"] = {
            "file": sidecar_file,
            "alignment": 16384,
            "size": size,
            "sha256": "e" * 64,
        }
    (root / "expert-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


# --- item 1: completeness is expert-aware -----------------------------------


def test_streamed_expert_model_missing_bank_is_not_complete(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model)

    assert cached_model_is_complete(model) is False
    reason = cached_model_incompleteness_reason(model)
    assert reason is not None
    assert "experts.bin" in reason and "missing" in reason


def test_streamed_expert_model_truncated_bank_is_not_complete(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model, size=64)
    (model / "experts.bin").write_bytes(b"\x00" * 10)

    reason = cached_model_incompleteness_reason(model)
    assert reason is not None
    assert "truncated" in reason
    assert "10 of 64" in reason


def test_streamed_expert_model_with_full_bank_is_complete(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model, size=64)
    (model / "experts.bin").write_bytes(b"\x00" * 64)

    assert cached_model_is_complete(model) is True
    assert cached_model_incompleteness_reason(model) is None


def test_oversized_bank_is_accepted(tmp_path: Path):
    """The manifest size is a floor, not an equality check."""

    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model, size=16)
    (model / "experts.bin").write_bytes(b"\x00" * 4096)

    assert cached_model_is_complete(model) is True


def test_manifest_without_sidecar_needs_no_bank(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model, sidecar_file=None)

    assert cached_model_is_complete(model) is True


def test_runtime_contract_declaring_experts_without_manifest_fails_closed(
    tmp_path: Path,
):
    model = _resident_model(tmp_path / "mtplx--experts")
    (model / "mtplx_runtime.json").write_text(
        json.dumps({"expert_manifest_file": "expert-manifest.json"}),
        encoding="utf-8",
    )

    reason = cached_model_incompleteness_reason(model)
    assert reason is not None
    assert "expert-manifest.json is missing" in reason


def test_unreadable_manifest_fails_closed(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    (model / "expert-manifest.json").write_text("{not json", encoding="utf-8")

    reason = cached_model_incompleteness_reason(model)
    assert reason is not None
    assert "could not be read" in reason


def test_manifest_naming_a_traversing_bank_is_rejected(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    _write_expert_manifest(model, sidecar_file="../../etc/passwd")

    reason = cached_model_incompleteness_reason(model)
    assert reason is not None
    assert "unsafe" in reason


def test_non_expert_model_is_unaffected(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--plain")

    assert cached_model_is_complete(model) is True
    assert expert_artifact_status(model)["streamed_experts"] is False


def test_sidecar_is_read_without_a_full_manifest_parse(tmp_path: Path):
    """Huge record lists must not have to be parsed to check the bank."""

    model = _resident_model(tmp_path / "mtplx--experts")
    payload = {
        "format": "mtplx-expert-manifest-v1",
        "records": [{"layer": i, "expert": i} for i in range(5000)],
        "sidecar": {
            "file": "experts.bin",
            "alignment": 16384,
            "size": 32,
            "sha256": "e" * 64,
        },
    }
    (model / "expert-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (model / "experts.bin").write_bytes(b"\x00" * 32)

    status = expert_artifact_status(model)
    assert status["ok"] is True
    assert status["sidecar_file"] == "experts.bin"
    assert status["expected_bytes"] == 32


def test_validate_mtplx_model_files_reports_missing_bank(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--experts")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text("{}", encoding="utf-8")
    (model / "mtp.safetensors").write_bytes(b"x")
    _write_expert_manifest(model)

    report = validate_mtplx_model_files(model)

    assert report["ok"] is False
    assert any("experts.bin" in item for item in report["missing_files"])
    assert report["expert_artifact"]["streamed_experts"] is True


def test_resolve_model_path_names_the_missing_bank(tmp_path: Path):
    model = _resident_model(tmp_path / "mtplx--example")
    _write_expert_manifest(model)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model_path("mtplx/example", cache_dir=tmp_path)

    assert "experts.bin" in str(excinfo.value)


# --- item 2: expert banks are fetched by default, opt-out is explicit -------


def test_repo_file_is_ignored_matches_banks_only():
    assert repo_file_is_ignored("experts.bin", EXPERT_BANK_IGNORE_PATTERNS) is True
    assert repo_file_is_ignored("sub/dir/experts.bin", EXPERT_BANK_IGNORE_PATTERNS) is True
    assert repo_file_is_ignored("model.safetensors", EXPERT_BANK_IGNORE_PATTERNS) is False
    assert repo_file_is_ignored("expert-manifest.json", EXPERT_BANK_IGNORE_PATTERNS) is False
    assert repo_file_is_ignored("experts.bin", ()) is False


def _install_fake_snapshot_download(monkeypatch, recorded: dict) -> None:
    def snapshot_download(**kwargs):
        recorded.update(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        _resident_model(destination)
        _write_expert_manifest(destination, size=32)
        if not kwargs.get("ignore_patterns"):
            (destination / "experts.bin").write_bytes(b"\x00" * 32)
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )


def test_pull_fetches_expert_banks_by_default(tmp_path: Path, monkeypatch):
    recorded: dict = {}
    _install_fake_snapshot_download(monkeypatch, recorded)

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert recorded["ignore_patterns"] is None
    assert result["partial_download"] is False
    assert result["excluded_patterns"] == []
    assert (Path(result["path"]) / "experts.bin").is_file()
    assert cached_model_is_complete(Path(result["path"])) is True


def test_pull_with_no_expert_banks_excludes_and_marks(tmp_path: Path, monkeypatch):
    recorded: dict = {}
    _install_fake_snapshot_download(monkeypatch, recorded)

    result = pull_model(
        "mtplx/example", cache_dir=tmp_path, include_expert_banks=False
    )
    path = Path(result["path"])

    assert recorded["ignore_patterns"] == list(EXPERT_BANK_IGNORE_PATTERNS)
    assert result["partial_download"] is True
    assert not (path / "experts.bin").exists()
    # The pull itself succeeds — a deliberate skip is not a download failure.
    assert (path / PARTIAL_DOWNLOAD_MARKER).is_file()


def test_deliberate_partial_download_is_not_reported_complete(
    tmp_path: Path, monkeypatch
):
    recorded: dict = {}
    _install_fake_snapshot_download(monkeypatch, recorded)
    result = pull_model(
        "mtplx/example", cache_dir=tmp_path, include_expert_banks=False
    )
    path = Path(result["path"])

    assert cached_model_is_complete(path) is False
    reason = cached_model_incompleteness_reason(path)
    assert reason is not None
    # Distinguishable from corruption: the message says it was on purpose.
    assert "on purpose" in reason
    assert "--no-expert-banks" in reason
    assert "truncated" not in reason
    assert partial_download_info(path)["excluded_patterns"] == list(
        EXPERT_BANK_IGNORE_PATTERNS
    )


def test_completing_a_partial_download_clears_the_marker(tmp_path: Path, monkeypatch):
    recorded: dict = {}
    _install_fake_snapshot_download(monkeypatch, recorded)
    pull_model("mtplx/example", cache_dir=tmp_path, include_expert_banks=False)

    result = pull_model("mtplx/example", cache_dir=tmp_path)
    path = Path(result["path"])

    assert not (path / PARTIAL_DOWNLOAD_MARKER).exists()
    assert result["partial_download"] is False
    assert cached_model_is_complete(path) is True


def test_a_partial_download_is_not_reused_as_a_cache_hit(tmp_path: Path, monkeypatch):
    recorded: dict = {}
    _install_fake_snapshot_download(monkeypatch, recorded)
    pull_model("mtplx/example", cache_dir=tmp_path, include_expert_banks=False)

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["reused_existing"] is False


def test_structured_progress_download_skips_ignored_files(tmp_path: Path, monkeypatch):
    from mtplx import hf_loader

    calls: list[str] = []

    def fake_download(repo_file, **kwargs):
        calls.append(repo_file.path)
        target = Path(kwargs["destination"]) / repo_file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * (repo_file.size_bytes or 1))
        return kwargs["last_emit_at"], kwargs["last_emit_size"]

    monkeypatch.setattr(hf_loader, "_download_repo_file", fake_download)
    monkeypatch.setattr(
        hf_loader,
        "_hub_runtime",
        lambda: (
            lambda: SimpleNamespace(
                model_info=lambda **_kw: SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="config.json", size=2),
                        SimpleNamespace(rfilename="expert-manifest.json", size=4),
                        SimpleNamespace(rfilename="experts.bin", size=4096),
                    ]
                )
            ),
            lambda **_kw: "https://example.invalid/f",
            lambda: SimpleNamespace(),
            lambda **_kw: {},
            lambda _r: None,
        ),
    )

    destination = tmp_path / "dest"
    destination.mkdir()
    _resolved, total, skipped = hf_loader._download_snapshot_with_structured_progress(
        repo_id="mtplx/example",
        revision=None,
        destination=destination,
        progress_callback=lambda _event: None,
        progress_interval_s=10.0,
        ignore_patterns=EXPERT_BANK_IGNORE_PATTERNS,
    )

    assert calls == ["config.json", "expert-manifest.json"]
    assert skipped == ["experts.bin"]
    # The excluded bank must not inflate the reported download total.
    assert total == 6
