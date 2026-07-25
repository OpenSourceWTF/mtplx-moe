from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from mtplx.hf_loader import (
    RepoFile,
    RepoInventory,
    _download_repo_file,
    cached_model_is_complete,
    cached_model_path,
    hf_token_for_download,
    hf_cache_report,
    list_cached_models,
    pull_model,
    remove_cached_model,
    repo_id_from_model_ref,
    resolve_model_path,
    safe_model_name,
    validate_mtplx_model_files,
)
from mtplx.profiles import DEFAULT_HF_MODEL_ID, LEGACY_OPTIMIZED_HF_MODEL_ID, QUALITY_HF_MODEL_ID


class _FakeHubResponse:
    def __init__(
        self,
        chunks: list[bytes | tuple[bytes, float]],
        status_code: int = 200,
        *,
        headers: dict[str, str] | None = None,
    ):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            delay = 0.0
            if isinstance(chunk, tuple):
                data, delay = chunk
            else:
                data = chunk
            if delay > 0:
                time.sleep(delay)
            yield data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHubSession:
    def __init__(self, files: dict[str, bytes | list[bytes | tuple[bytes, float]]]):
        self.files = files
        self.requests: list[dict[str, object]] = []

    def get(self, url: str, **kwargs):
        filename = url.removeprefix("fake://")
        self.requests.append({"url": url, **kwargs})
        payload = self.files[filename]
        chunks: list[bytes | tuple[bytes, float]]
        if isinstance(payload, bytes):
            chunks = [payload]
        else:
            chunks = payload
        return _FakeHubResponse(chunks)


def _install_fake_hub(
    monkeypatch,
    files: dict[str, bytes | list[bytes | tuple[bytes, float]]],
    *,
    captured: dict[str, object] | None = None,
    resolved_revision: str = "c" * 40,
    lfs_overrides: dict[str, str] | None = None,
) -> _FakeHubSession:
    captured = captured if captured is not None else {}
    session = _FakeHubSession(files)
    hub = ModuleType("huggingface_hub")
    hub.__path__ = []

    class FakeHfApi:
        def model_info(self, **kwargs):
            captured["model_info_token"] = kwargs.get("token")
            captured["model_info_calls"] = (
                int(captured.get("model_info_calls", 0)) + 1
            )
            return SimpleNamespace(
                sha=resolved_revision,
                siblings=[
                    SimpleNamespace(
                        rfilename=name,
                        size=sum(
                            len(item[0] if isinstance(item, tuple) else item)
                            for item in payload
                        )
                        if isinstance(payload, list)
                        else len(payload),
                        lfs=SimpleNamespace(
                            sha256=(lfs_overrides or {}).get(
                                name,
                                hashlib.sha256(
                                    b"".join(
                                        item[0] if isinstance(item, tuple) else item
                                        for item in payload
                                    )
                                    if isinstance(payload, list)
                                    else payload
                                ).hexdigest(),
                            )
                        ),
                    )
                    for name, payload in files.items()
                ],
            )

    def fake_hf_hub_url(*, repo_id, filename, revision=None):
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        captured.setdefault("download_revisions", []).append(revision)
        return f"fake://{filename}"

    hub.HfApi = FakeHfApi
    hub.hf_hub_url = fake_hf_hub_url
    hub.get_session = lambda: session
    hub.snapshot_download = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("structured progress should not use snapshot_download")
    )

    utils = ModuleType("huggingface_hub.utils")

    def fake_build_hf_headers(**kwargs):
        captured["headers_token"] = kwargs.get("token")
        return {}

    def fake_hf_raise_for_status(response):
        response.raise_for_status()

    utils.build_hf_headers = fake_build_hf_headers
    utils.hf_raise_for_status = fake_hf_raise_for_status
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return session


class _RangeSession:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int,
        content_range: str | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content_range = content_range
        self.requests: list[dict[str, object]] = []

    def get(self, url: str, **kwargs):
        self.requests.append({"url": url, **kwargs})
        headers = (
            {"Content-Range": self.content_range}
            if self.content_range is not None
            else None
        )
        return _FakeHubResponse(
            [self.payload],
            status_code=self.status_code,
            headers=headers,
        )


def _download_one(
    tmp_path: Path,
    repo_file: RepoFile,
    *,
    payload: bytes,
    status_code: int,
    content_range: str | None = None,
):
    session = _RangeSession(
        payload,
        status_code=status_code,
        content_range=content_range,
    )
    seen_revisions: list[str | None] = []
    result = _download_repo_file(
        repo_file,
        repo_id="owner/model",
        revision="d" * 40,
        destination=tmp_path,
        session=session,
        hf_hub_url=lambda **kwargs: (
            seen_revisions.append(kwargs.get("revision")) or "fake://file"
        ),
        build_hf_headers=lambda **_kwargs: {},
        hf_raise_for_status=lambda response: response.raise_for_status(),
        callback=None,
        total_bytes=repo_file.size_bytes,
        started_at=time.monotonic(),
        progress_interval_s=0,
        last_emit_at=time.monotonic(),
        last_emit_size=0,
    )
    return result, session, seen_revisions


def _write_partial_metadata(
    root: Path,
    repo_file: RepoFile,
    *,
    revision: str = "d" * 40,
) -> None:
    assert repo_file.sha256 is not None
    (root / f"{repo_file.path}.incomplete.meta.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "repo_id": "owner/model",
                "revision": revision,
                "path": repo_file.path,
                "size_bytes": repo_file.size_bytes,
                "validator": f"sha256:{repo_file.sha256}",
            }
        ),
        encoding="utf-8",
    )


def test_download_repo_file_hashes_resumed_prefix_and_appends_206(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    partial = tmp_path / "experts.bin.incomplete"
    partial.write_bytes(complete[:3])
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )
    _write_partial_metadata(tmp_path, repo_file)

    (_emit_at, _emit_size, digest), session, revisions = _download_one(
        tmp_path,
        repo_file,
        payload=complete[3:],
        status_code=206,
        content_range="bytes 3-5/6",
    )

    assert (tmp_path / "experts.bin").read_bytes() == complete
    assert session.requests[0]["headers"]["Range"] == "bytes=3-"
    assert digest.sha256 == repo_file.sha256
    assert revisions == ["d" * 40]


def test_download_repo_file_resets_prefix_and_hasher_on_range_200(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    (tmp_path / "experts.bin.incomplete").write_bytes(complete[:3])
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )
    _write_partial_metadata(tmp_path, repo_file)

    (_emit_at, _emit_size, digest), _session, _revisions = _download_one(
        tmp_path,
        repo_file,
        payload=complete,
        status_code=200,
    )

    assert (tmp_path / "experts.bin").read_bytes() == complete
    assert digest.sha256 == repo_file.sha256


def test_download_repo_file_installs_exact_partial_on_range_416(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    (tmp_path / "experts.bin.incomplete").write_bytes(complete)
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )
    _write_partial_metadata(tmp_path, repo_file)

    (_emit_at, _emit_size, digest), _session, _revisions = _download_one(
        tmp_path,
        repo_file,
        payload=b"",
        status_code=416,
    )

    assert (tmp_path / "experts.bin").read_bytes() == complete
    assert digest.sha256 == repo_file.sha256


def test_download_repo_file_rejects_lfs_digest_mismatch_before_install(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _download_one(
            tmp_path,
            repo_file,
            payload=complete,
            status_code=200,
        )

    assert not (tmp_path / "experts.bin").exists()


def test_download_repo_file_rejects_intermediate_symlink_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "nested").symlink_to(outside, target_is_directory=True)
    repo_file = RepoFile(path="nested/experts.bin", size_bytes=1, sha256=None)

    with pytest.raises(RuntimeError, match="symlink"):
        _download_one(
            tmp_path,
            repo_file,
            payload=b"x",
            status_code=200,
        )

    assert not (outside / "experts.bin").exists()


def test_download_repo_file_rejects_mismatched_content_range(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    (tmp_path / "experts.bin.incomplete").write_bytes(complete[:3])
    (tmp_path / "experts.bin.incomplete.meta.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "repo_id": "owner/model",
                "revision": "d" * 40,
                "path": "experts.bin",
                "size_bytes": len(complete),
                "validator": f"sha256:{hashlib.sha256(complete).hexdigest()}",
            }
        ),
        encoding="utf-8",
    )
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )

    with pytest.raises(RuntimeError, match="Content-Range"):
        _download_one(
            tmp_path,
            repo_file,
            payload=complete[3:],
            status_code=206,
            content_range="bytes 0-2/6",
        )

    assert not (tmp_path / "experts.bin").exists()


def test_download_repo_file_never_resumes_digestless_partial_or_accepts_416(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "config.json.incomplete"
    partial.write_bytes(b"old")
    repo_file = RepoFile(path="config.json", size_bytes=3, sha256=None)

    with pytest.raises(RuntimeError, match="HTTP 416"):
        _download_one(
            tmp_path,
            repo_file,
            payload=b"",
            status_code=416,
        )

    assert not (tmp_path / "config.json").exists()


def test_download_repo_file_discards_cross_revision_partial_metadata(
    tmp_path: Path,
) -> None:
    complete = b"NEWNEW"
    (tmp_path / "experts.bin.incomplete").write_bytes(b"old")
    (tmp_path / "experts.bin.incomplete.meta.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "path": "experts.bin",
                "size_bytes": len(complete),
                "validator": f"sha256:{hashlib.sha256(complete).hexdigest()}",
            }
        ),
        encoding="utf-8",
    )
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )

    (_emit_at, _emit_size, _digest), session, _revisions = _download_one(
        tmp_path,
        repo_file,
        payload=complete,
        status_code=200,
    )

    assert "Range" not in session.requests[0]["headers"]
    assert (tmp_path / "experts.bin").read_bytes() == complete


def test_verified_final_removes_stale_partial_and_metadata(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    (tmp_path / "experts.bin").write_bytes(complete)
    (tmp_path / "experts.bin.incomplete").write_bytes(b"stale")
    (tmp_path / "experts.bin.incomplete.meta.json").write_text(
        "{}",
        encoding="utf-8",
    )
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )

    _download_one(
        tmp_path,
        repo_file,
        payload=b"",
        status_code=200,
    )

    assert not (tmp_path / "experts.bin.incomplete").exists()
    assert not (tmp_path / "experts.bin.incomplete.meta.json").exists()


def test_downloader_replaces_final_without_mutating_retained_inode(
    tmp_path: Path,
) -> None:
    old = b"OLDOLD"
    complete = b"NEWNEW"
    target = tmp_path / "experts.bin"
    target.write_bytes(old)
    retained = os.open(target, os.O_RDONLY)
    repo_file = RepoFile(
        path="experts.bin",
        size_bytes=len(complete),
        sha256=hashlib.sha256(complete).hexdigest(),
    )
    try:
        _download_one(
            tmp_path,
            repo_file,
            payload=complete,
            status_code=200,
        )

        assert os.pread(retained, len(old), 0) == old
        assert target.read_bytes() == complete
    finally:
        os.close(retained)


def test_pull_model_serializes_same_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

    def inventory(_repo_id: str, *, revision: str | None = None):
        assert revision is not None
        return RepoInventory(
            resolved_revision=revision,
            files=(
                RepoFile(path="config.json", size_bytes=3),
                RepoFile(path="model.safetensors", size_bytes=7),
            ),
        )

    def snapshot_download(**kwargs):
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.05)
            destination = Path(kwargs["local_dir"])
            (destination / "config.json").write_text("{}\n", encoding="utf-8")
            (destination / "model.safetensors").write_bytes(b"weights")
            return str(destination)
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr("mtplx.hf_loader._query_repo_inventory", inventory)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                pull_model,
                "owner/model",
                cache_dir=tmp_path,
                revision=revision,
            )
            for revision in ("a" * 40, "b" * 40)
        ]
        for future in futures:
            future.result()

    assert peak_active == 1


def test_pull_model_lfs_mismatch_never_emits_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    files = {
        "config.json": b"{}\n",
        "model.safetensors": b"weights",
    }
    _install_fake_hub(
        monkeypatch,
        files,
        lfs_overrides={"model.safetensors": "0" * 64},
    )
    events: list[dict] = []

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pull_model(
            "owner/model",
            cache_dir=tmp_path,
            progress_callback=events.append,
            progress_interval_s=0,
        )

    assert "complete" not in [event["event"] for event in events]


def test_pull_model_requires_model_info_immutable_sha(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors": b"weights",
        },
        resolved_revision="main",
    )
    events: list[dict] = []

    with pytest.raises(RuntimeError, match="immutable commit SHA"):
        pull_model(
            "owner/model",
            cache_dir=tmp_path,
            progress_callback=events.append,
        )

    assert events == []


def test_pull_model_plain_no_callback_keeps_snapshot_downloader_but_pins_sha(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolved_revision = "e" * 40
    captured: dict[str, object] = {}
    hub = ModuleType("huggingface_hub")
    hub.__path__ = []

    class FakeHfApi:
        def model_info(self, **_kwargs):
            return SimpleNamespace(
                sha=resolved_revision,
                siblings=[
                    SimpleNamespace(
                        rfilename="config.json",
                        size=3,
                        lfs=None,
                    ),
                    SimpleNamespace(
                        rfilename="model.safetensors",
                        size=7,
                        lfs=None,
                    ),
                ],
            )

    def snapshot_download(**kwargs):
        captured.update(kwargs)
        destination = Path(kwargs["local_dir"])
        (destination / "config.json").write_text("{}\n", encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    hub.HfApi = FakeHfApi
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    result = pull_model("owner/plain", cache_dir=tmp_path)

    assert captured["revision"] == resolved_revision
    assert result["resolved_revision"] == resolved_revision
    assert result["expert_admission"] is None


def test_repo_id_from_model_ref_accepts_hf_url_and_repo_id():
    assert repo_id_from_model_ref("mtplx/example") == "mtplx/example"
    assert (
        repo_id_from_model_ref("https://huggingface.co/mtplx/example/tree/main")
        == "mtplx/example"
    )
    assert repo_id_from_model_ref("models/local-model") is None


def test_repo_id_from_model_ref_maps_known_public_aliases():
    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Quality") == QUALITY_HF_MODEL_ID
    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Speed") == DEFAULT_HF_MODEL_ID
    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized") == LEGACY_OPTIMIZED_HF_MODEL_ID


def test_known_public_alias_wins_over_bare_cwd_folder(tmp_path: Path, monkeypatch):
    (tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality").mkdir()
    monkeypatch.chdir(tmp_path)

    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Quality") == QUALITY_HF_MODEL_ID
    assert repo_id_from_model_ref("./Qwen3.6-27B-MTPLX-Optimized-Quality") is None


def test_safe_model_name_and_cache_path(tmp_path: Path):
    assert safe_model_name("mtplx/example") == "mtplx--example"
    assert cached_model_path("mtplx/example", cache_dir=tmp_path) == tmp_path / "mtplx--example"


def test_resolve_model_path_uses_cache_for_hf_refs(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"1234")

    assert resolve_model_path("mtplx/example", cache_dir=tmp_path) == cached


def test_cached_model_is_complete_rejects_interrupted_indexed_download(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_rejects_partial_index_even_with_one_shard(
    tmp_path: Path,
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {'
        '"a": "model-00001-of-00002.safetensors", '
        '"b": "model-00002-of-00002.safetensors"'
        '}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_rejects_incomplete_transfer_marker(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")
    (cached / "model.safetensors.incomplete").write_bytes(b"partial")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_rejects_shards_that_sort_before_index(
    tmp_path: Path,
):
    # Interrupted pull of Qwen/Qwen3.5-122B-A10B: shard names like
    # "model.safetensors-00001-of-00039.safetensors" download before
    # "model.safetensors.index.json", so a cancel leaves complete shards,
    # no index, and no .incomplete marker.
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors-00001-of-00039.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_accepts_single_file_model(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is True


def test_pull_model_reuses_complete_destination_without_redownload(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    def fail_snapshot_download(**_kwargs):
        raise AssertionError("complete cached model should not download again")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_snapshot_download),
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["path"] == str(cached)
    assert result["reused_existing"] is True
    assert result["resumed_existing"] is False


def test_pull_model_resumes_incomplete_destination(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    download_cache = cached / ".cache" / "huggingface" / "download"
    download_cache.mkdir(parents=True)
    (download_cache / "model-00001-of-00001.safetensors.incomplete").write_bytes(
        b"partial"
    )
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["path"] == str(cached)
    assert result["reused_existing"] is False
    assert result["resumed_existing"] is True
    assert result["started_size_bytes"] > 0
    assert events[0]["event"] == "resume"
    assert "progress" in [event["event"] for event in events]
    assert [event["event"] for event in events[-2:]] == ["verifying", "complete"]


def test_pull_model_resumes_qwen_mtplx_folder_missing_required_sidecars(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / safe_model_name(QUALITY_HF_MODEL_ID)
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "tokenizer.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
            "mtp.safetensors": b"mtp",
            "mtplx_runtime.json": b"{}\n",
        },
    )
    events: list[dict] = []

    result = pull_model(
        QUALITY_HF_MODEL_ID,
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["reused_existing"] is False
    assert result["resumed_existing"] is True
    assert result["validation"]["ok"] is True
    assert [event["event"] for event in events[-2:]] == ["verifying", "complete"]


def test_pull_model_structured_stream_reports_written_bytes(
    tmp_path: Path, monkeypatch
):
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": [
                (b"a" * 16, 0.02),
                (b"a" * 48, 0.02),
            ],
        },
    )
    events: list[dict] = []

    pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0.01,
    )

    progress_events = [event for event in events if event["event"] == "progress"]
    assert any(event.get("delta_bytes", 0) > 0 for event in progress_events)
    assert all(event.get("message") == "Downloading model files" for event in progress_events)


def test_hf_token_for_download_uses_explicit_env_only(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    assert hf_token_for_download() is False

    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    assert hf_token_for_download() == "hf_explicit"


def test_pull_model_downloads_public_models_anonymously_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured: dict[str, object] = {}
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
        captured=captured,
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["path"] == str(tmp_path / "mtplx--example")
    assert result["resolved_revision"] == "c" * 40
    assert captured["model_info_token"] is False
    assert captured["model_info_calls"] == 1
    assert captured["headers_token"] is False
    assert set(captured["download_revisions"]) == {"c" * 40}
    assert events[0]["total_bytes"] == 81


def test_resolve_model_path_reports_missing_cache(tmp_path: Path):
    try:
        resolve_model_path("mtplx/example", cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "mtplx pull mtplx/example" in str(exc)
    else:
        raise AssertionError("expected missing cache error")


def test_resolve_model_path_rejects_missing_local_path(tmp_path: Path):
    missing = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality"
    try:
        resolve_model_path(str(missing), cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "not available locally" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected missing local path error")


def test_list_and_remove_cached_models(tmp_path: Path):
    (tmp_path / ".tmp").mkdir()
    model = tmp_path / "mtplx--example"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text("{}\n", encoding="utf-8")
    (model / "small.bin").write_bytes(b"1234")

    rows = list_cached_models(cache_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0].repo_id == "mtplx/example"
    assert rows[0].has_config is True
    assert rows[0].has_runtime_contract is True
    assert rows[0].validation["missing_files"]
    assert rows[0].to_dict()["recommended_profile"] is None
    assert rows[0].size_bytes >= 4

    removed = remove_cached_model("mtplx/example", cache_dir=tmp_path)
    assert removed["removed"] is True
    assert not model.exists()


def test_hf_cache_report_is_no_network(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    cache = tmp_path / "missing-cache"
    report = hf_cache_report(cache_dir=cache)

    assert report["cache_dir"] == str(cache)
    assert report["cache_exists"] is False
    assert report["cached_models"] == 0
    assert "token_present" in report
    assert "disk_free_bytes" in report


def test_validate_mtplx_model_files_reports_required_payload(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
    ):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text('{"arch_id": "qwen3-next-mtp"}\n', encoding="utf-8")

    validation = validate_mtplx_model_files(model)

    assert validation["ok"] is True
    assert validation["missing_files"] == []
    assert validation["contract_arch_id"] == "qwen3-next-mtp"


def test_validate_mtplx_model_files_accepts_configured_nested_mtp_sidecar(tmp_path: Path):
    model = tmp_path / "model"
    (model / "mtp").mkdir(parents=True)
    (model / "config.json").write_text(
        '{"mlx_lm_extra_tensors": {"mtp_file": "mtp/weights.safetensors"}}\n',
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "model.safetensors.index.json"):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text('{"arch_id": "qwen3-next-mtp"}\n', encoding="utf-8")
    (model / "mtp" / "weights.safetensors").write_bytes(b"mtp")

    validation = validate_mtplx_model_files(model)

    assert validation["ok"] is True
    assert validation["missing_files"] == []
    assert validation["mtp_sidecar_candidates"][0] == "mtp/weights.safetensors"


def _write_complete_single(root: Path, shards: int = 1) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    weight_map = {
        f"w{i}": f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    for name in set(weight_map.values()):
        (root / name).write_bytes(b"weights")


def test_cached_model_is_complete_accepts_assistant_pair_bundle(tmp_path: Path):
    # QA-112: Gemma-4 pair bundles have no top-level config.json; the old
    # check failed them at 100% with "weight shards missing".
    bundle = tmp_path / "Youssofal--Gemma4-MTPLX-Optimized-Speed"
    bundle.mkdir()
    (bundle / "mtplx_pair.json").write_text(
        json.dumps({"layout": {"target": "target", "assistant": "assistant"}}),
        encoding="utf-8",
    )
    _write_complete_single(bundle / "target", shards=4)
    _write_complete_single(bundle / "assistant", shards=1)

    assert cached_model_is_complete(bundle) is True


def test_cached_model_is_complete_rejects_pair_bundle_missing_assistant_shard(
    tmp_path: Path,
):
    bundle = tmp_path / "Youssofal--Gemma4-MTPLX-Optimized-Speed"
    bundle.mkdir()
    (bundle / "mtplx_pair.json").write_text(
        json.dumps({"layout": {"target": "target", "assistant": "assistant"}}),
        encoding="utf-8",
    )
    _write_complete_single(bundle / "target", shards=4)
    # assistant half: index references a shard that never downloaded.
    assistant = bundle / "assistant"
    assistant.mkdir()
    (assistant / "config.json").write_text("{}\n", encoding="utf-8")
    (assistant / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors"}}), encoding="utf-8"
    )

    assert cached_model_is_complete(bundle) is False
