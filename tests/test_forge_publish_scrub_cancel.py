"""`forge publish`: metadata scrubbing on upload, and cancel-marker honoring."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.cli import main
from mtplx.commands import forge


def _runtime_with_local_paths() -> dict:
    return {
        "arch_id": "hy_v3",
        "mtplx_version": "2.0.2",
        "forge_provenance": {
            "forged_locally": True,
            "source_repo": "tencent/Hy3",
            "intended_hf_repo": "davidtai/hy3-q4-mlx-mtp",
            "forge_inputs": {
                "source_path": "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
                "output_path": "/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp",
            },
        },
    }


class _RecordingApi:
    def __init__(self) -> None:
        self.folder_kwargs: dict = {}
        self.files: list[dict] = []

    def create_repo(self, **kwargs):
        return None

    def upload_folder(self, **kwargs):
        self.folder_kwargs = kwargs
        return SimpleNamespace(oid="rev-folder")

    def upload_file(self, **kwargs):
        self.files.append(kwargs)
        return SimpleNamespace(oid="rev-file")

    def model_info(self, repo_id, *, token=None):
        return SimpleNamespace(sha="rev-final")


def _make_model(tmp_path: Path) -> Path:
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    (local / "mtplx_runtime.json").write_text(
        json.dumps(_runtime_with_local_paths()), encoding="utf-8"
    )
    return local


def _publish_argv(local: Path, tmp_path: Path, run_id: str = "p1") -> list[str]:
    return [
        "forge",
        "publish",
        "--path",
        str(local),
        "--repo",
        "owner/Fixture-MTPLX-Speed",
        "--visibility",
        "private",
        "--license",
        "apache-2.0",
        "--out",
        str(tmp_path / "publish"),
        "--run-id",
        run_id,
        "--token",
        "stdin",
    ]


@pytest.fixture
def cancel_dir(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "cancel"
    monkeypatch.setenv("MTPLX_FORGE_CANCEL_DIR", str(root))
    return root


def test_published_runtime_is_scrubbed_and_local_copy_is_untouched(
    tmp_path, monkeypatch, cancel_dir
):
    local = _make_model(tmp_path)
    api = _RecordingApi()
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\n"))

    code = main(_publish_argv(local, tmp_path))

    assert code == 0
    # The raw runtime never reaches the folder commit.
    assert api.folder_kwargs["ignore_patterns"] == ["mtplx_runtime.json"]
    uploaded = [f for f in api.files if f["path_in_repo"] == "mtplx_runtime.json"]
    assert len(uploaded) == 1

    published = json.loads(Path(uploaded[0]["path_or_fileobj"]).read_text())
    provenance = published["forge_provenance"]
    assert "/Users/" not in json.dumps(published)
    assert "intended_hf_repo" not in provenance
    assert provenance["source_repo"] == "tencent/Hy3"

    # The operator's own artifact keeps its full provenance.
    local_runtime = json.loads((local / "mtplx_runtime.json").read_text())
    assert (
        local_runtime["forge_provenance"]["forge_inputs"]["source_path"]
        == "/Users/davidtai/.cache/huggingface/hy3-mtp-layer80"
    )
    assert local_runtime["forge_provenance"]["intended_hf_repo"] == (
        "davidtai/hy3-q4-mlx-mtp"
    )


def test_publish_without_a_runtime_contract_uploads_the_whole_folder(
    tmp_path, monkeypatch, cancel_dir
):
    local = tmp_path / "model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    api = _RecordingApi()
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\n"))

    assert main(_publish_argv(local, tmp_path)) == 0
    assert "ignore_patterns" not in api.folder_kwargs
    assert api.files == []


def test_publish_stops_before_creating_the_repo_when_cancelled(
    tmp_path, monkeypatch, cancel_dir
):
    local = _make_model(tmp_path)

    class ExplodingApi(_RecordingApi):
        def create_repo(self, **kwargs):
            raise AssertionError("cancelled publish must not create a repo")

    monkeypatch.setattr(forge, "_make_hf_api", lambda: ExplodingApi())
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\n"))

    cancel_dir.mkdir(parents=True, exist_ok=True)
    (cancel_dir / "p1.json").write_text('{"run_id": "p1"}', encoding="utf-8")

    assert main(_publish_argv(local, tmp_path)) == 130


def test_publish_stops_after_the_folder_upload_when_cancelled_mid_flight(
    tmp_path, monkeypatch, cancel_dir
):
    """A cancel landing during upload_folder takes effect when it returns."""

    local = _make_model(tmp_path)

    class CancellingApi(_RecordingApi):
        def upload_folder(self, **kwargs):
            (cancel_dir / "p1.json").write_text('{"run_id": "p1"}', encoding="utf-8")
            return super().upload_folder(**kwargs)

    api = CancellingApi()
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\n"))
    cancel_dir.mkdir(parents=True, exist_ok=True)

    assert main(_publish_argv(local, tmp_path)) == 130
    # It stopped before the runtime/README follow-up commits.
    assert api.files == []
    publish_json = json.loads(
        (tmp_path / "publish" / "p1" / "publish.json").read_text(encoding="utf-8")
    )
    assert publish_json.get("finished") is not True


def test_cancel_marker_is_scoped_to_its_run_id(tmp_path, monkeypatch, cancel_dir):
    local = _make_model(tmp_path)
    api = _RecordingApi()
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\n"))

    cancel_dir.mkdir(parents=True, exist_ok=True)
    (cancel_dir / "other-run.json").write_text('{"run_id": "other"}', encoding="utf-8")

    assert main(_publish_argv(local, tmp_path, run_id="p1")) == 0
