"""Hugging Face model resolution and local cache helpers."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from mtplx.artifacts import _hf_repo_id_from_ref
from mtplx.expert_admission import (
    TrustedFileDigest,
    admit_expert_artifact,
    ensure_expert_admitted,
    load_valid_admission_receipt,
)
from mtplx.models.laguna_config import (
    LAGUNA_S_2_1_REPO_ID,
    LAGUNA_S_2_1_REPO_BYTES,
    LAGUNA_S_2_1_REQUIRED_FILES,
    LAGUNA_S_2_1_REVISION,
    laguna_s_2_1_artifact_integrity_errors,
)
from mtplx.profiles import DEFAULT_PROFILE_NAME


DEFAULT_MODEL_CACHE = Path("~/.mtplx/models").expanduser()
DownloadProgressCallback = Callable[[dict[str, Any]], None]
REQUIRED_MTPLX_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "mtplx_runtime.json",
)
MTP_SIDECAR_FALLBACKS = (
    "mtp.safetensors",
    "mtp/weights.safetensors",
    "model-mtp.safetensors",
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
EXPERT_MANIFEST_FILE = "expert-manifest.json"
MAX_RUNTIME_CONTRACT_BYTES = 1024 * 1024
SOURCE_MARKER_FILE = ".mtplx-source.json"


def read_bounded_artifact_member(
    root: Path | str,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read a bounded artifact member through the trusted descriptor path.

    Members may be regular files inside ``root`` or standard Hugging Face
    ``snapshots/<revision> -> ../../blobs/<digest>`` links. The resolver
    rejects every other escape, then the resolved target is opened with
    ``O_NOFOLLOW`` and checked through its descriptor before any bytes are
    consumed.
    """

    from mtplx.expert_manifest import (
        ExpertManifestError,
        _read_file_nofollow,
        resolve_artifact_member,
    )

    try:
        resolved = resolve_artifact_member(Path(root), name)
        return _read_file_nofollow(resolved, max_bytes=max_bytes)
    except ExpertManifestError:
        raise
    except OSError as exc:
        raise ExpertManifestError(
            f"could not read artifact member {name}: {exc}"
        ) from exc


@dataclass(frozen=True)
class RepoFile:
    path: str
    size_bytes: int | None
    sha256: str | None = None


@dataclass(frozen=True)
class RepoInventory:
    resolved_revision: str
    files: tuple[RepoFile, ...]


def _effective_model_revision(repo_id: str, revision: str | None) -> str | None:
    if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
        if revision is not None and revision != LAGUNA_S_2_1_REVISION:
            raise ValueError(
                "Laguna-S-2.1 support is pinned to revision "
                f"{LAGUNA_S_2_1_REVISION}"
            )
        return LAGUNA_S_2_1_REVISION
    return revision


def _source_marker_matches(
    destination: Path,
    *,
    repo_id: str,
    revision: str | None,
) -> bool:
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return True
    try:
        payload = json.loads(
            (destination / SOURCE_MARKER_FILE).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return payload == {"repo_id": repo_id, "revision": revision}


def _write_source_marker(
    destination: Path,
    *,
    repo_id: str,
    revision: str | None,
) -> None:
    (destination / SOURCE_MARKER_FILE).write_text(
        json.dumps(
            {"repo_id": repo_id, "revision": revision},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_pinned_laguna_files(destination: Path, repo_id: str) -> None:
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return
    missing_or_wrong = laguna_s_2_1_artifact_integrity_errors(destination)
    if missing_or_wrong:
        raise RuntimeError(
            "pinned Laguna snapshot is incomplete or differs from revision "
            f"{LAGUNA_S_2_1_REVISION}: "
            + ", ".join(sorted(missing_or_wrong))
        )


def _pull_validation(path: Path, repo_id: str) -> dict[str, Any]:
    validation = validate_mtplx_model_files(path)
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return validation
    _validate_pinned_laguna_files(path, repo_id)
    return {
        **validation,
        "ok": True,
        "missing_files": [],
        "contract_error": None,
        "required_files": sorted(LAGUNA_S_2_1_REQUIRED_FILES),
        "mtp_supported": False,
        "runtime_compatibility": "native-ar-only",
    }


def _require_download_disk_headroom(
    root: Path,
    *,
    total_bytes: int | None,
    started_size_bytes: int,
) -> None:
    if total_bytes is None or total_bytes <= 0:
        return
    remaining = max(0, int(total_bytes) - max(0, int(started_size_bytes)))
    headroom = 5 * 1024**3
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError:
        return
    required = remaining + headroom
    if free < required:
        raise RuntimeError(
            "insufficient free disk space for model download: "
            f"need {required / 1024**3:.1f} GiB including headroom, "
            f"have {free / 1024**3:.1f} GiB"
        )


def _repo_file_from_sibling(sibling: Any) -> RepoFile | None:
    name = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
    if not isinstance(name, str) or not name.strip():
        return None
    size = getattr(sibling, "size", None)
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        digest = lfs.get("sha256")
    else:
        digest = getattr(lfs, "sha256", None)
    sha256 = (
        digest.lower()
        if isinstance(digest, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
        else None
    )
    return RepoFile(
        path=name,
        size_bytes=size if isinstance(size, int) else None,
        sha256=sha256,
    )


def _repo_inventory_from_info(info: Any, repo_id: str) -> RepoInventory:
    resolved_revision = getattr(info, "sha", None)
    if (
        not isinstance(resolved_revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", resolved_revision) is None
    ):
        raise RuntimeError(
            f"Hugging Face repo {repo_id} did not resolve to an immutable commit SHA"
        )
    files = tuple(
        repo_file
        for sibling in (getattr(info, "siblings", None) or [])
        if (repo_file := _repo_file_from_sibling(sibling)) is not None
    )
    if not files:
        raise RuntimeError(
            f"Hugging Face repo {repo_id} did not return downloadable files."
        )
    return RepoInventory(
        resolved_revision=resolved_revision.lower(),
        files=files,
    )


def _query_repo_inventory(
    repo_id: str,
    *,
    revision: str | None = None,
) -> RepoInventory:
    try:
        hf_hub = importlib.import_module("huggingface_hub")
        api = hf_hub.HfApi()
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=hf_token_for_download(),
        )
        return _repo_inventory_from_info(info, repo_id)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(_classify_pull_error(exc, repo_id)) from exc


def _query_repo_files(repo_id: str, *, revision: str | None = None) -> list[RepoFile]:
    """Return downloadable files with Hub-reported sizes when available."""

    try:
        return list(_query_repo_inventory(repo_id, revision=revision).files)
    except RuntimeError:
        return []


def _query_repo_total_bytes(repo_id: str, *, revision: str | None = None) -> int | None:
    """Best-effort estimate of the remote repo's total size."""

    total = 0
    for repo_file in _query_repo_files(repo_id, revision=revision):
        if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes > 0:
            total += repo_file.size_bytes
    return total or None


@contextlib.contextmanager
def _suppress_hf_hub_progress() -> Iterator[None]:
    """Suppress Hugging Face tqdm bars while MTPLX owns download progress."""

    previous_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    disabled_via_helper = False
    try:
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            disabled_via_helper = True
        except Exception:
            pass
        yield
    finally:
        if disabled_via_helper:
            try:
                from huggingface_hub.utils import enable_progress_bars

                enable_progress_bars()
            except Exception:
                pass
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_env


def model_cache_dir(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MODEL_CACHE


def safe_model_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "--")


def repo_id_from_model_ref(value: str) -> str | None:
    return _hf_repo_id_from_ref(value)


def cached_model_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path:
    return model_cache_dir(cache_dir) / safe_model_name(repo_id)


def hf_token_for_download() -> str | bool:
    """Use explicit env auth only; public pulls should never need HF login."""

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False


def _complete_indexed_weights(path: Path, index_name: str) -> bool:
    index = path / index_name
    if not index.is_file():
        return False
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return False
    filenames = {
        name
        for name in weight_map.values()
        if isinstance(name, str) and name.strip()
    }
    if not filenames:
        return False
    for name in filenames:
        shard = path / name
        try:
            if not shard.is_file() or shard.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


_SHARD_FILENAME_RE = re.compile(r"-\d+-of-\d+", re.IGNORECASE)


def _has_incomplete_transfers(path: Path) -> bool:
    """``snapshot_download`` stages in-flight files as ``*.incomplete``.

    Markers inside the hub's ``.cache`` bookkeeping tree are ignored: they
    can outlive a successful resume, and the weight checks verify the final
    files directly. A marker next to the weights, however, means the final
    file never landed.
    """

    try:
        for marker in path.rglob("*.incomplete"):
            if ".cache" in marker.relative_to(path).parts:
                continue
            return True
    except OSError:
        pass
    return False


def _complete_unindexed_weights(path: Path) -> bool:
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        for candidate in path.glob(pattern):
            try:
                if not candidate.is_file() or candidate.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            # A shard-named file implies a weight index the download has not
            # reached yet; shard names can sort before the index (e.g.
            # "model.safetensors-00001-of-00039.safetensors" precedes
            # "model.safetensors.index.json"), so treat the copy as partial
            # rather than as a complete single-file model.
            if _SHARD_FILENAME_RE.search(candidate.name):
                return False
            return True
    return False


def _declares_streamed_experts(path: Path) -> bool:
    """Whether the runtime contract says this artifact streams its experts."""

    from mtplx.expert_manifest import ExpertManifestError

    try:
        payload = read_bounded_artifact_member(
            path,
            "mtplx_runtime.json",
            max_bytes=MAX_RUNTIME_CONTRACT_BYTES,
        )
        contract = json.loads(payload)
    except (
        ExpertManifestError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False
    if not isinstance(contract, dict):
        return False
    if contract.get("expert_manifest_file"):
        return True
    streaming = contract.get("expert_streaming")
    if isinstance(streaming, dict):
        return bool(streaming)
    return bool(streaming)


def expert_artifact_status(path: Path) -> dict[str, Any]:
    """Validate a complete authoritative streamed-expert artifact.

    This is a construction-time trust boundary. Manifest identity and geometry
    go through the runtime's authoritative parser/spec validator, then every
    declared bank part is resolved inside the artifact (including supported
    repository-local Hugging Face blob links) and checked through an opened
    descriptor. No generation-path work is added.
    """

    manifest_path = path / EXPERT_MANIFEST_FILE
    try:
        os.lstat(manifest_path)
        manifest_present = True
    except FileNotFoundError:
        manifest_present = False
    except OSError:
        # An unreadable directory entry is present for trust-boundary
        # purposes: validation below must fail closed rather than silently
        # treating the cache as an ordinary non-streamed model.
        manifest_present = True
    status: dict[str, Any] = {
        "streamed_experts": manifest_present or _declares_streamed_experts(path),
        "manifest_file": EXPERT_MANIFEST_FILE,
        "manifest_present": manifest_present,
        "sidecar_file": None,
        "sidecar_files": [],
        "expected_bytes": None,
        "actual_bytes": None,
        "ok": True,
        "reason": None,
    }
    if not status["streamed_experts"]:
        return status
    if not manifest_present:
        status["ok"] = False
        status["reason"] = (
            f"{EXPERT_MANIFEST_FILE} is missing but the artifact declares streamed experts"
        )
        return status

    try:
        from mtplx.expert_manifest import (
            MAX_MANIFEST_BYTES,
            ExpertManifest,
            ExpertManifestError,
            _load_json_bytes,
            resolve_artifact_member,
            validate_expert_manifest_spec,
        )
        from mtplx.expert_streaming_models import get_model_spec

        payload = read_bounded_artifact_member(
            path,
            EXPERT_MANIFEST_FILE,
            max_bytes=MAX_MANIFEST_BYTES,
        )
        manifest = ExpertManifest.from_dict(
            _load_json_bytes(payload, source=str(manifest_path))
        )
        spec = get_model_spec(manifest.model_key)
        validate_expert_manifest_spec(manifest, spec)
        if manifest.sidecar is None:
            raise ExpertManifestError(
                "authoritative streamed manifest requires sidecar metadata"
            )
        sidecar_shards = tuple(
            shard for shard in manifest.shards if shard.kind == "sidecar"
        )
        part_names = tuple(part.file for part in manifest.sidecar.parts)
        if (
            len(sidecar_shards) != len(manifest.sidecar.parts)
            or {shard.name for shard in sidecar_shards} != set(part_names)
        ):
            raise ExpertManifestError(
                "manifest is not authoritative for every sidecar part"
            )

        status["sidecar_files"] = list(part_names)
        status["sidecar_file"] = part_names[0] if len(part_names) == 1 else None
        status["expected_bytes"] = sum(
            part.size for part in manifest.sidecar.parts
        )
        actual_bytes = 0
        for part in manifest.sidecar.parts:
            try:
                resolved = resolve_artifact_member(path, part.file)
            except ExpertManifestError as exc:
                raise ExpertManifestError(
                    f"expert bank part {part.file} is missing or unsafe ({exc})"
                ) from exc
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(resolved, flags)
            except OSError as exc:
                raise ExpertManifestError(
                    f"could not open expert bank part {part.file}: {exc}"
                ) from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ExpertManifestError(
                        f"expert bank part {part.file} is not a regular file"
                    )
                actual = metadata.st_size
            finally:
                os.close(descriptor)
            if actual < part.size:
                raise ExpertManifestError(
                    f"expert bank part {part.file} is truncated "
                    f"({actual} of {part.size} bytes)"
                )
            actual_bytes += actual
        status["actual_bytes"] = actual_bytes
    except (OSError, TypeError, ValueError) as exc:
        status["ok"] = False
        status["reason"] = (
            f"{EXPERT_MANIFEST_FILE} failed authoritative validation: {exc}"
        )
    return status


def cached_model_is_complete(path: Path) -> bool:
    """Return whether a Hub cache directory is ready to run.

    ``snapshot_download(local_dir=...)`` creates the destination early. An
    interrupted pull can therefore leave config/tokenizer files plus an index,
    which looks cached even though the weight shards are missing.
    """

    if not path.is_dir():
        return False
    if _has_incomplete_transfers(path):
        return False
    # Assistant-pair bundles (Gemma 4) have no top-level config.json — the
    # weights live under target/ and assistant/ with an mtplx_pair.json
    # marker. Require both halves to be complete (QA-112).
    if (path / "mtplx_pair.json").is_file():
        return _pair_bundle_is_complete(path)
    if not (path / "config.json").is_file():
        return False
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    if any((path / name).is_file() for name in index_names):
        weights_complete = any(
            _complete_indexed_weights(path, name) for name in index_names
        )
    else:
        weights_complete = _complete_unindexed_weights(path)
    if not weights_complete:
        return False
    expert_status = expert_artifact_status(path)
    return not (
        expert_status["streamed_experts"]
        and not expert_status["ok"]
    )


def _pair_bundle_is_complete(path: Path) -> bool:
    """Completeness for an assistant-pair bundle (target/ + assistant/).

    Resolves the half-paths from the pair marker's declared layout and
    checks each half exactly the way a single model is checked.
    """

    try:
        from mtplx.gemma4_pair import resolve_gemma4_pair_paths
    except Exception:
        return False
    resolved = resolve_gemma4_pair_paths(path)
    if not resolved:
        return False
    for key in ("target_model", "assistant_model"):
        half = resolved.get(key)
        if not half or not cached_model_is_complete(Path(half)):
            return False
    return True


def _repo_requires_qwen_mtplx_payload(repo_id: str) -> bool:
    lower = repo_id.lower()
    return lower.startswith("youssofal/qwen3.") and "mtplx" in lower


def _cached_model_ready_for_repo(path: Path, repo_id: str) -> bool:
    if not cached_model_is_complete(path):
        return False
    if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
        if not _source_marker_matches(
            path,
            repo_id=repo_id,
            revision=LAGUNA_S_2_1_REVISION,
        ):
            return False
        try:
            _validate_pinned_laguna_files(path, repo_id)
        except RuntimeError:
            return False
    if _repo_requires_qwen_mtplx_payload(repo_id):
        return bool(validate_mtplx_model_files(path).get("ok"))
    return True


def resolve_model_path(model_ref: str, *, cache_dir: str | Path | None = None) -> Path:
    local = Path(model_ref).expanduser()
    if local.exists():
        return local
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise FileNotFoundError(f"Model path is not available locally: {local}")
    cached = cached_model_path(repo_id, cache_dir=cache_dir)
    if cached.is_dir():
        expert_status = expert_artifact_status(cached)
        if (
            expert_status["streamed_experts"]
            and not expert_status["ok"]
        ):
            raise FileNotFoundError(
                f"Cached streamed model {repo_id} has invalid expert artifacts: "
                f"{expert_status['reason']}. Run: mtplx pull {repo_id}"
            )
    if _cached_model_ready_for_repo(cached, repo_id):
        return cached
    raise FileNotFoundError(
        f"Model {repo_id} is not cached. Run: mtplx pull {repo_id}"
    )


def _configured_mtp_file(contract: dict[str, Any] | None, config: dict[str, Any] | None) -> str | None:
    for source in (config, contract):
        extra = source.get("mlx_lm_extra_tensors", {}) if isinstance(source, dict) else {}
        if isinstance(extra, dict) and extra.get("mtp_file"):
            return str(extra["mtp_file"])
    if isinstance(contract, dict):
        for key in ("mtp_file", "mtp_sidecar_file"):
            value = contract.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _mtp_sidecar_candidates(path: Path, contract: dict[str, Any] | None = None) -> list[str]:
    config: dict[str, Any] | None = None
    config_path = path / "config.json"
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else None
        except Exception:
            config = None
    candidates: list[str] = []
    configured = _configured_mtp_file(contract, config)
    if configured:
        candidates.append(configured)
    candidates.extend(MTP_SIDECAR_FALLBACKS)
    result: list[str] = []
    for rel in candidates:
        if rel not in result:
            result.append(rel)
    return result


def _mtp_sidecar_exists(path: Path, contract: dict[str, Any] | None = None) -> bool:
    for rel in _mtp_sidecar_candidates(path, contract):
        try:
            if (path / rel).is_file():
                return True
        except OSError:
            continue
    return False


def validate_mtplx_model_files(path: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_MTPLX_MODEL_FILES if not (path / name).exists()]
    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    contract_path = path / "mtplx_runtime.json"
    if contract_path.exists():
        try:
            loaded = json.loads(contract_path.read_text(encoding="utf-8"))
            contract = loaded if isinstance(loaded, dict) else None
        except Exception as exc:
            contract_error = str(exc)
    sidecar_candidates = _mtp_sidecar_candidates(path, contract)
    if not _mtp_sidecar_exists(path, contract):
        missing.append("mtp sidecar")
    return {
        "ok": not missing and contract_error is None,
        "required_files": list(REQUIRED_MTPLX_MODEL_FILES) + [sidecar_candidates[0]],
        "mtp_sidecar_candidates": sidecar_candidates,
        "missing_files": missing,
        "contract_present": contract_path.exists(),
        "contract_arch_id": contract.get("arch_id") if isinstance(contract, dict) else None,
        "contract_error": contract_error,
    }


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _emit_download_progress(callback: DownloadProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # Progress reporting must never break a model download.
        return


def _hub_runtime() -> tuple[Any, Callable[..., str], Callable[[], Any], Callable[..., dict[str, str]], Callable[[Any], None]]:
    """Import the small Hub surface used by the app installer.

    Tests often patch ``huggingface_hub`` with a lightweight module object, so
    this helper keeps the imports forgiving while still using the official Hub
    helpers when they are available.
    """

    try:
        hf_hub = importlib.import_module("huggingface_hub")
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for mtplx pull: {exc}") from exc

    try:
        from huggingface_hub.utils import build_hf_headers, hf_raise_for_status
    except Exception:

        def build_hf_headers(**_kwargs: Any) -> dict[str, str]:
            return {}

        def hf_raise_for_status(response: Any) -> None:
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            elif int(getattr(response, "status_code", 200)) >= 400:
                raise RuntimeError(f"Hugging Face request failed with HTTP {response.status_code}")

    required = ["hf_hub_url", "get_session", "HfApi"]
    missing = [name for name in required if not hasattr(hf_hub, name)]
    if missing:
        raise RuntimeError(
            "huggingface_hub is too old for structured mtplx pull "
            f"(missing {', '.join(missing)})"
        )
    return (
        hf_hub.HfApi,
        hf_hub.hf_hub_url,
        hf_hub.get_session,
        build_hf_headers,
        hf_raise_for_status,
    )


def _classify_pull_error(exc: BaseException, repo_id: str) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {401, 403}:
        return (
            f"Hugging Face denied access to {repo_id}. "
            "Sign in with an access token or request access, then retry."
        )
    if status_code == 404:
        return f"Hugging Face could not find {repo_id}. Check the model name, then retry."
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "Not enough disk space to finish the model download. Free space, then retry."
    return str(exc)


def _safe_repo_relative_path(repo_file: RepoFile) -> PurePosixPath:
    relative = PurePosixPath(repo_file.path)
    if (
        not repo_file.path
        or "\\" in repo_file.path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError(
            f"unsafe file path in Hugging Face repo: {repo_file.path}"
        )
    return relative


def _safe_destination_for_repo_file(destination: Path, repo_file: RepoFile) -> Path:
    relative = _safe_repo_relative_path(repo_file)
    return destination.joinpath(*relative.parts)


@contextlib.contextmanager
def _open_safe_repo_parent(
    destination: Path,
    repo_file: RepoFile,
) -> Iterator[tuple[int, Path, str]]:
    """Open every parent with ``openat`` so symlink swaps cannot escape."""

    relative = _safe_repo_relative_path(repo_file)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    parent = destination
    try:
        try:
            descriptor = os.open(destination, directory_flags)
        except OSError as exc:
            raise RuntimeError(
                f"model download directory must be a real directory, not a symlink: "
                f"{destination} ({exc})"
            ) from exc
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"download path contains an unsafe intermediate symlink "
                    f"or non-directory member: {parent / component} ({exc})"
                ) from exc
            os.close(descriptor)
            descriptor = child_descriptor
            parent /= component
        yield descriptor, parent, relative.name
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _regular_file_metadata(path: Path) -> os.stat_result | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"could not inspect download file {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"download file must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"download file is not a regular file: {path}")
    return metadata


def _regular_file_metadata_at(
    directory_descriptor: int,
    name: str,
    display_path: Path,
) -> os.stat_result | None:
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(
            f"could not inspect download file {display_path}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"download file must not be a symlink: {display_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(
            f"download file is not a regular file: {display_path}"
        )
    return metadata


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_descriptor(
    descriptor: int,
    digest: Any,
    *,
    limit: int | None = None,
) -> int:
    total = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while limit is None or total < limit:
        size = DOWNLOAD_CHUNK_SIZE if limit is None else min(
            DOWNLOAD_CHUNK_SIZE, limit - total
        )
        try:
            chunk = os.read(descriptor, size)
        except InterruptedError:
            continue
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return total


def _trusted_file_digest(
    descriptor: int,
    digest: Any,
) -> TrustedFileDigest:
    metadata = os.fstat(descriptor)
    return TrustedFileDigest(
        sha256=digest.hexdigest(),
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size,
        st_mtime_ns=metadata.st_mtime_ns,
        st_ctime_ns=metadata.st_ctime_ns,
    )


def _rehash_trusted_descriptor(
    descriptor: int,
    *,
    display_path: str,
) -> TrustedFileDigest:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    read_bytes = _hash_descriptor(descriptor, digest)
    after = os.fstat(descriptor)
    if (
        read_bytes != before.st_size
        or _file_identity(before) != _file_identity(after)
    ):
        raise RuntimeError(
            f"download file changed while hashing installed bytes: {display_path}"
        )
    return TrustedFileDigest(
        sha256=digest.hexdigest(),
        st_dev=after.st_dev,
        st_ino=after.st_ino,
        st_size=after.st_size,
        st_mtime_ns=after.st_mtime_ns,
        st_ctime_ns=after.st_ctime_ns,
    )


def _hash_existing_file(
    path: Path,
    *,
    directory_descriptor: int | None = None,
    name: str | None = None,
) -> TrustedFileDigest:
    open_kwargs: dict[str, Any] = {}
    open_target: Path | str = path
    if directory_descriptor is not None and name is not None:
        open_kwargs["dir_fd"] = directory_descriptor
        open_target = name
    descriptor = os.open(
        open_target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        **open_kwargs,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"download file is not a regular file: {path}")
        digest = hashlib.sha256()
        read_bytes = _hash_descriptor(descriptor, digest)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if read_bytes != before.st_size or _file_identity(before) != _file_identity(after):
        raise RuntimeError(f"download file changed while hashing: {path}")
    return TrustedFileDigest(
        sha256=digest.hexdigest(),
        st_dev=after.st_dev,
        st_ino=after.st_ino,
        st_size=after.st_size,
        st_mtime_ns=after.st_mtime_ns,
        st_ctime_ns=after.st_ctime_ns,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while downloading model file")
        view = view[written:]


def _partial_metadata_payload(
    repo_file: RepoFile,
    *,
    repo_id: str,
    revision: str | None,
) -> dict[str, Any] | None:
    if repo_file.sha256 is None:
        return None
    return {
        "schema": 1,
        "repo_id": repo_id,
        "revision": revision,
        "path": repo_file.path,
        "size_bytes": repo_file.size_bytes,
        "validator": f"sha256:{repo_file.sha256}",
    }


def _read_partial_metadata_at(
    directory_descriptor: int,
    metadata_name: str,
) -> dict[str, Any] | None:
    try:
        descriptor = os.open(
            metadata_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024:
            return None
        payload = bytearray()
        while len(payload) <= 64 * 1024:
            chunk = os.read(descriptor, min(8192, 64 * 1024 + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > 64 * 1024:
            return None
        decoded = json.loads(payload.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        os.close(descriptor)


def _write_partial_metadata_at(
    directory_descriptor: int,
    metadata_name: str,
    payload: dict[str, Any],
) -> None:
    temporary_name = metadata_name + ".tmp"
    try:
        os.unlink(temporary_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        _write_all(
            descriptor,
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(
        temporary_name,
        metadata_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)


def _remove_partial_state_at(
    directory_descriptor: int,
    partial_name: str,
    metadata_name: str,
) -> None:
    removed = False
    for name in (partial_name, metadata_name, metadata_name + ".tmp"):
        try:
            os.unlink(name, dir_fd=directory_descriptor)
            removed = True
        except FileNotFoundError:
            continue
    if removed:
        os.fsync(directory_descriptor)


def _response_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return value if isinstance(value, str) else None


def _validate_content_range(
    response: Any,
    *,
    expected_start: int,
    expected_size: int | None,
) -> None:
    value = _response_header(response, "Content-Range")
    match = (
        re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", value)
        if value is not None
        else None
    )
    if match is None:
        raise RuntimeError("invalid or missing Content-Range for resumed download")
    start, end = int(match.group(1)), int(match.group(2))
    if start != expected_start or end < start:
        raise RuntimeError(
            f"Content-Range does not match requested offset {expected_start}: {value}"
        )
    if (
        expected_size is not None
        and (
            match.group(3) == "*"
            or int(match.group(3)) != expected_size
        )
    ):
        raise RuntimeError(
            f"Content-Range does not match expected size {expected_size}: {value}"
        )


def _emit_current_download_size(
    callback: DownloadProgressCallback | None,
    *,
    repo_id: str,
    destination: Path,
    total_bytes: int | None,
    started_at: float,
    last_emit_at: float,
    last_emit_size: int,
    file_path: str | None = None,
) -> tuple[float, int]:
    now = time.monotonic()
    current_size = directory_size_bytes(destination)
    interval = max(0.001, now - last_emit_at)
    delta = current_size - last_emit_size
    _emit_download_progress(
        callback,
        {
            "event": "progress",
            "repo_id": repo_id,
            "path": str(destination),
            "file": file_path,
            "size_bytes": current_size,
            "total_bytes": total_bytes,
            "delta_bytes": delta,
            "rate_bps": float(max(0, delta)) / interval,
            "elapsed_s": now - started_at,
            "interval_s": interval,
            "stalled_s": 0,
            "message": "Downloading model files",
        },
    )
    return now, current_size


def _open_hub_stream(session: Any, url: str, headers: dict[str, str]) -> Any:
    stream = getattr(session, "stream", None)
    if callable(stream):
        return stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=60,
        )
    return session.get(url, headers=headers, stream=True, timeout=(10, 60))


def _iter_response_bytes(response: Any) -> Iterator[bytes]:
    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        yield from iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE)
        return
    iter_bytes = getattr(response, "iter_bytes", None)
    if callable(iter_bytes):
        yield from iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE)
        return
    raise RuntimeError("Hugging Face response does not support byte streaming")


def _download_repo_file_at(
    repo_file: RepoFile,
    *,
    directory_descriptor: int,
    parent: Path,
    target_name: str,
    repo_id: str,
    revision: str | None,
    destination: Path,
    session: Any,
    hf_hub_url: Callable[..., str],
    build_hf_headers: Callable[..., dict[str, str]],
    hf_raise_for_status: Callable[[Any], None],
    callback: DownloadProgressCallback | None,
    total_bytes: int | None,
    started_at: float,
    progress_interval_s: float,
    last_emit_at: float,
    last_emit_size: int,
) -> tuple[float, int, TrustedFileDigest]:
    target = parent / target_name
    expected_size = repo_file.size_bytes
    partial_name = target_name + ".incomplete"
    partial = parent / partial_name
    partial_metadata_name = partial_name + ".meta.json"
    expected_partial_metadata = _partial_metadata_payload(
        repo_file,
        repo_id=repo_id,
        revision=revision,
    )
    target_metadata = _regular_file_metadata_at(
        directory_descriptor,
        target_name,
        target,
    )
    partial_metadata = _regular_file_metadata_at(
        directory_descriptor,
        partial_name,
        partial,
    )

    # Installed paths are immutable once admission can retain their inode.
    # A refresh never opens ``target`` for writing: it stages a separate
    # partial and atomically replaces the pathname after verification.
    if (
        target_metadata is not None
        and expected_size is not None
        and target_metadata.st_size == expected_size
        and repo_file.sha256 is not None
    ):
        existing_digest = _hash_existing_file(
            target,
            directory_descriptor=directory_descriptor,
            name=target_name,
        )
        if existing_digest.sha256 == repo_file.sha256:
            _remove_partial_state_at(
                directory_descriptor,
                partial_name,
                partial_metadata_name,
            )
            emitted_at, emitted_size = _emit_current_download_size(
                callback,
                repo_id=repo_id,
                destination=destination,
                total_bytes=total_bytes,
                started_at=started_at,
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
                file_path=repo_file.path,
            )
            return emitted_at, emitted_size, existing_digest
    # A nonmatching target stays linked and available to its existing receipt
    # and any retained readers until the new partial has been fully downloaded,
    # verified, and fsynced. ``install_verified_partial`` is the only refresh
    # path that replaces it.
    recorded_partial_metadata = _read_partial_metadata_at(
        directory_descriptor,
        partial_metadata_name,
    )
    resumable = (
        expected_partial_metadata is not None
        and recorded_partial_metadata == expected_partial_metadata
    )
    if partial_metadata is not None and not resumable:
        _remove_partial_state_at(
            directory_descriptor,
            partial_name,
            partial_metadata_name,
        )
        partial_metadata = None
    elif partial_metadata is None and recorded_partial_metadata is not None:
        _remove_partial_state_at(
            directory_descriptor,
            partial_name,
            partial_metadata_name,
        )
    existing = partial_metadata.st_size if partial_metadata is not None else 0
    if expected_size is not None and existing > expected_size:
        _remove_partial_state_at(
            directory_descriptor,
            partial_name,
            partial_metadata_name,
        )
        existing = 0
    if expected_partial_metadata is not None:
        _write_partial_metadata_at(
            directory_descriptor,
            partial_metadata_name,
            expected_partial_metadata,
        )
        resumable = True
    else:
        resumable = False

    descriptor = os.open(
        partial_name,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        prefix_before = os.fstat(descriptor)
        if not stat.S_ISREG(prefix_before.st_mode):
            raise RuntimeError(f"partial download is not a regular file: {partial}")
        existing = prefix_before.st_size
        digest = hashlib.sha256()
        if existing:
            hashed = _hash_descriptor(descriptor, digest, limit=existing)
            prefix_after = os.fstat(descriptor)
            if (
                hashed != existing
                or _file_identity(prefix_before) != _file_identity(prefix_after)
            ):
                raise RuntimeError(
                    f"partial download changed while hashing: {repo_file.path}"
                )

        def install_verified_partial() -> TrustedFileDigest:
            os.fsync(descriptor)
            final_metadata = os.fstat(descriptor)
            if (
                expected_size is not None
                and final_metadata.st_size != expected_size
            ):
                raise RuntimeError(
                    f"incomplete download for {repo_file.path}: "
                    f"expected {expected_size} bytes, got {final_metadata.st_size}"
                )
            trusted = _trusted_file_digest(descriptor, digest)
            if (
                repo_file.sha256 is not None
                and trusted.sha256 != repo_file.sha256
            ):
                _remove_partial_state_at(
                    directory_descriptor,
                    partial_name,
                    partial_metadata_name,
                )
                raise RuntimeError(
                    f"SHA-256 mismatch for {repo_file.path}: "
                    f"expected {repo_file.sha256}, got {trusted.sha256}"
                )
            os.replace(
                partial_name,
                target_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.fsync(directory_descriptor)
            _remove_partial_state_at(
                directory_descriptor,
                partial_name,
                partial_metadata_name,
            )
            after_install = os.fstat(descriptor)
            if (
                trusted.st_dev,
                trusted.st_ino,
                trusted.st_size,
                trusted.st_mtime_ns,
                trusted.st_ctime_ns,
            ) != _file_identity(after_install):
                trusted = _rehash_trusted_descriptor(
                    descriptor,
                    display_path=repo_file.path,
                )
                if (
                    repo_file.sha256 is not None
                    and trusted.sha256 != repo_file.sha256
                ):
                    raise RuntimeError(
                        f"SHA-256 mismatch for installed {repo_file.path}: "
                        f"expected {repo_file.sha256}, got {trusted.sha256}"
                    )
            return trusted

        headers = build_hf_headers(token=hf_token_for_download())
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        url = hf_hub_url(
            repo_id=repo_id,
            filename=repo_file.path,
            revision=revision,
        )
        response_stream = _open_hub_stream(session, url, headers)
        with response_stream as response:
            status_code = int(getattr(response, "status_code", 200))
            if status_code == 206:
                if not resumable or repo_file.sha256 is None:
                    raise RuntimeError(
                        "unexpected Content-Range without a strongly validated "
                        "resumable partial"
                    )
                _validate_content_range(
                    response,
                    expected_start=existing,
                    expected_size=expected_size,
                )
            if existing > 0 and status_code == 200:
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                existing = 0
            elif (
                existing > 0
                and status_code == 416
                and resumable
                and repo_file.sha256 is not None
                and expected_size is not None
                and existing == expected_size
            ):
                trusted = install_verified_partial()
                emitted_at, emitted_size = _emit_current_download_size(
                    callback,
                    repo_id=repo_id,
                    destination=destination,
                    total_bytes=total_bytes,
                    started_at=started_at,
                    last_emit_at=last_emit_at,
                    last_emit_size=last_emit_size,
                    file_path=repo_file.path,
                )
                return emitted_at, emitted_size, trusted
            hf_raise_for_status(response)
            os.lseek(descriptor, 0, os.SEEK_END)
            for chunk in _iter_response_bytes(response):
                if not chunk:
                    continue
                _write_all(descriptor, chunk)
                digest.update(chunk)
                now = time.monotonic()
                if now - last_emit_at >= progress_interval_s:
                    last_emit_at, last_emit_size = _emit_current_download_size(
                        callback,
                        repo_id=repo_id,
                        destination=destination,
                        total_bytes=total_bytes,
                        started_at=started_at,
                        last_emit_at=last_emit_at,
                        last_emit_size=last_emit_size,
                        file_path=repo_file.path,
                    )
        trusted = install_verified_partial()
    finally:
        os.close(descriptor)

    emitted_at, emitted_size = _emit_current_download_size(
        callback,
        repo_id=repo_id,
        destination=destination,
        total_bytes=total_bytes,
        started_at=started_at,
        last_emit_at=last_emit_at,
        last_emit_size=last_emit_size,
        file_path=repo_file.path,
    )
    return emitted_at, emitted_size, trusted


def _download_repo_file(
    repo_file: RepoFile,
    *,
    repo_id: str,
    revision: str | None,
    destination: Path,
    session: Any,
    hf_hub_url: Callable[..., str],
    build_hf_headers: Callable[..., dict[str, str]],
    hf_raise_for_status: Callable[[Any], None],
    callback: DownloadProgressCallback | None,
    total_bytes: int | None,
    started_at: float,
    progress_interval_s: float,
    last_emit_at: float,
    last_emit_size: int,
) -> tuple[float, int, TrustedFileDigest]:
    with _open_safe_repo_parent(
        destination,
        repo_file,
    ) as (directory_descriptor, parent, target_name):
        return _download_repo_file_at(
            repo_file,
            directory_descriptor=directory_descriptor,
            parent=parent,
            target_name=target_name,
            repo_id=repo_id,
            revision=revision,
            destination=destination,
            session=session,
            hf_hub_url=hf_hub_url,
            build_hf_headers=build_hf_headers,
            hf_raise_for_status=hf_raise_for_status,
            callback=callback,
            total_bytes=total_bytes,
            started_at=started_at,
            progress_interval_s=progress_interval_s,
            last_emit_at=last_emit_at,
            last_emit_size=last_emit_size,
        )


def _download_snapshot_with_structured_progress(
    *,
    repo_id: str,
    inventory: RepoInventory,
    destination: Path,
    progress_callback: DownloadProgressCallback | None,
    progress_interval_s: float,
) -> tuple[Path, int | None, dict[str, TrustedFileDigest]]:
    (
        _HfApi,
        hf_hub_url,
        get_session,
        build_hf_headers,
        hf_raise_for_status,
    ) = _hub_runtime()
    repo_files = inventory.files

    total_bytes = sum(
        repo_file.size_bytes
        for repo_file in repo_files
        if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes > 0
    ) or None
    session = get_session()
    started_at = time.monotonic()
    last_emit_at = started_at
    last_emit_size = directory_size_bytes(destination)
    downloaded_digests: dict[str, TrustedFileDigest] = {}
    for repo_file in repo_files:
        try:
            last_emit_at, last_emit_size, digest = _download_repo_file(
                repo_file,
                repo_id=repo_id,
                revision=inventory.resolved_revision,
                destination=destination,
                session=session,
                hf_hub_url=hf_hub_url,
                build_hf_headers=build_hf_headers,
                hf_raise_for_status=hf_raise_for_status,
                callback=progress_callback,
                total_bytes=total_bytes,
                started_at=started_at,
                progress_interval_s=max(0.1, progress_interval_s),
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
            )
            downloaded_digests[repo_file.path] = digest
        except Exception as exc:
            raise RuntimeError(_classify_pull_error(exc, repo_id)) from exc
    return destination, total_bytes, downloaded_digests


@dataclass(frozen=True)
class CachedModel:
    repo_id: str
    path: Path
    size_bytes: int
    has_runtime_contract: bool
    has_config: bool
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1_000_000_000, 3),
            "has_runtime_contract": self.has_runtime_contract,
            "has_config": self.has_config,
            "validation": self.validation,
            "recommended_profile": DEFAULT_PROFILE_NAME if self.validation.get("ok") else None,
            "delete_command": f"mtplx remove {self.repo_id}",
        }


def list_cached_models(*, cache_dir: str | Path | None = None) -> list[CachedModel]:
    root = model_cache_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[CachedModel] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        repo_id = child.name.replace("--", "/")
        rows.append(
            CachedModel(
                repo_id=repo_id,
                path=child,
                size_bytes=directory_size_bytes(child),
                has_runtime_contract=(child / "mtplx_runtime.json").exists(),
                has_config=(child / "config.json").exists(),
                validation=validate_mtplx_model_files(child),
            )
        )
    return rows


def _local_matches_remote_index(
    path: Path, repo_id: str, revision: str | None
) -> bool:
    """Best-effort remote freshness check for an explicit pull.

    A pull is a stated intent to sync, so a locally-complete copy must
    still pick up files added to the repo after the first download
    (restored vision towers, repaired contracts). The weight index is
    the cheap proxy: when it changed upstream, fall through to the
    download branch and let snapshot_download fetch only the delta.
    Network failures err on reuse so offline pulls keep working.
    """

    local_index = path / "model.safetensors.index.json"
    if not local_index.is_file():
        return True
    try:
        from huggingface_hub import hf_hub_download

        remote = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            revision=revision,
        )
        return Path(remote).read_bytes() == local_index.read_bytes()
    except Exception:
        return True


def _repo_is_exact_streaming_catalog(repo_id: str) -> bool:
    try:
        from mtplx.default_models import streaming_catalog_models

        return repo_id.casefold() in {
            model.hf_model_id.casefold()
            for model in streaming_catalog_models()
        }
    except Exception:
        return False


def _inventory_declares_experts(inventory: RepoInventory) -> bool:
    return any(repo_file.path == EXPERT_MANIFEST_FILE for repo_file in inventory.files)


def _is_immutable_hub_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None
    )


_DESTINATION_LOCKS_GUARD = threading.Lock()
_DESTINATION_LOCKS: dict[str, threading.Lock] = {}


@contextlib.contextmanager
def _destination_pull_lock(destination: Path) -> Iterator[None]:
    key = str(destination.resolve())
    with _DESTINATION_LOCKS_GUARD:
        thread_lock = _DESTINATION_LOCKS.setdefault(key, threading.Lock())
    lock_root = destination.parent / ".locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{destination.name}.lock"
    with thread_lock:
        descriptor = os.open(
            lock_path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def pull_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    progress_callback: DownloadProgressCallback | None = None,
    progress_interval_s: float = 10.0,
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(
            f"pull requires a Hugging Face repo id or URL, got: {model_ref}"
        )
    destination = cached_model_path(repo_id, cache_dir=cache_dir)
    with _destination_pull_lock(destination):
        return _pull_model_unlocked(
            model_ref,
            cache_dir=cache_dir,
            revision=revision,
            progress_callback=progress_callback,
            progress_interval_s=progress_interval_s,
        )


def _pull_model_unlocked(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    progress_callback: DownloadProgressCallback | None = None,
    progress_interval_s: float = 10.0,
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(f"pull requires a Hugging Face repo id or URL, got: {model_ref}")
    revision = _effective_model_revision(repo_id, revision)
    root = model_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = cached_model_path(repo_id, cache_dir=root)

    started_size = directory_size_bytes(destination)
    local_ready = (
        destination.exists()
        and _cached_model_ready_for_repo(destination, repo_id)
    )
    local_expert_status = (
        expert_artifact_status(destination)
        if destination.is_dir()
        else {"streamed_experts": False, "ok": True}
    )
    local_streamed = bool(local_expert_status.get("streamed_experts"))
    exact_streaming_repo = _repo_is_exact_streaming_catalog(repo_id)
    inventory: RepoInventory | None = None
    inventory_error: RuntimeError | None = None
    try:
        inventory = _query_repo_inventory(repo_id, revision=revision)
    except RuntimeError as exc:
        inventory_error = exc

    resolved_revision = (
        inventory.resolved_revision if inventory is not None else revision
    )
    remote_streamed = (
        _inventory_declares_experts(inventory)
        if inventory is not None
        else False
    )
    expert_repo = exact_streaming_repo or local_streamed or remote_streamed
    admission: dict[str, Any] | None = None
    reuse_allowed = False

    if local_ready and expert_repo:
        if inventory is not None:
            receipt = load_valid_admission_receipt(
                destination,
                revision=inventory.resolved_revision,
            )
            reuse_allowed = (
                receipt is not None
                and receipt.get("repo_id") == repo_id
            )
        else:
            # Offline expert reuse is allowed only when a prior admission is
            # bound to an immutable Hub SHA. A mutable requested revision or
            # an identity-only cache check is insufficient.
            receipt = load_valid_admission_receipt(destination)
            receipt_revision = (
                receipt.get("revision") if receipt is not None else None
            )
            reuse_allowed = (
                receipt is not None
                and receipt.get("repo_id") == repo_id
                and _is_immutable_hub_revision(receipt_revision)
                and (
                    revision is None
                    or (
                        _is_immutable_hub_revision(revision)
                        and str(receipt_revision).casefold()
                        == str(revision).casefold()
                    )
                )
            )
            if reuse_allowed:
                resolved_revision = receipt_revision
    elif local_ready and not expert_repo:
        reuse_allowed = (
            _source_marker_matches(
                destination,
                repo_id=repo_id,
                revision=revision,
            )
            and _local_matches_remote_index(
                destination,
                repo_id,
                resolved_revision,
            )
        )

    if reuse_allowed:
        resolved = destination
        reused_existing = True
        resumed_existing = False
        validation = validate_mtplx_model_files(resolved)
        _validate_pinned_laguna_files(resolved, repo_id)
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "cached MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
        if expert_repo:
            _emit_download_progress(
                progress_callback,
                {
                    "event": "verifying",
                    "repo_id": repo_id,
                    "path": str(resolved),
                    "size_bytes": directory_size_bytes(resolved),
                    "total_bytes": directory_size_bytes(resolved),
                },
            )
            admission = ensure_expert_admitted(
                resolved,
                repo_id=repo_id,
                revision=resolved_revision,
            )
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": directory_size_bytes(resolved),
                "total_bytes": directory_size_bytes(resolved),
                "delta_bytes": 0,
                "reused_existing": True,
            },
        )
    else:
        if inventory is None:
            if inventory_error is not None:
                raise inventory_error
            raise RuntimeError(
                f"could not resolve immutable Hugging Face metadata for {repo_id}"
            )
        reused_existing = False
        resumed_existing = destination.exists() and started_size > 0
        total_bytes = sum(
            repo_file.size_bytes
            for repo_file in inventory.files
            if isinstance(repo_file.size_bytes, int)
            and repo_file.size_bytes > 0
        ) or None
        if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
            total_bytes = LAGUNA_S_2_1_REPO_BYTES
        _require_download_disk_headroom(
            root,
            total_bytes=total_bytes,
            started_size_bytes=started_size,
        )
        destination.mkdir(parents=True, exist_ok=True)
        _emit_download_progress(
            progress_callback,
            {
                "event": "resume" if resumed_existing else "start",
                "repo_id": repo_id,
                "path": str(destination),
                "size_bytes": started_size,
                "total_bytes": total_bytes,
            },
        )
        progress_suppression = (
            _suppress_hf_hub_progress()
            if progress_callback is not None
            else contextlib.nullcontext()
        )
        downloaded_digests: dict[str, TrustedFileDigest] = {}
        with progress_suppression:
            if progress_callback is not None or expert_repo:
                (
                    resolved,
                    total_bytes_from_download,
                    downloaded_digests,
                ) = _download_snapshot_with_structured_progress(
                    repo_id=repo_id,
                    inventory=inventory,
                    destination=destination,
                    progress_callback=progress_callback,
                    progress_interval_s=progress_interval_s,
                )
                if total_bytes_from_download:
                    total_bytes = total_bytes_from_download
            else:
                try:
                    from huggingface_hub import snapshot_download
                except Exception as exc:
                    raise RuntimeError(
                        f"huggingface_hub is required for mtplx pull: {exc}"
                    ) from exc
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=inventory.resolved_revision,
                    local_dir=str(destination),
                    token=hf_token_for_download(),
                )
                resolved = Path(path)
        _emit_download_progress(
            progress_callback,
            {
                "event": "verifying",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": directory_size_bytes(resolved),
                "total_bytes": total_bytes,
            },
        )
        validation = validate_mtplx_model_files(resolved)
        if not cached_model_is_complete(resolved):
            expert_status = expert_artifact_status(resolved)
            if (
                expert_status["streamed_experts"]
                and not expert_status["ok"]
            ):
                raise RuntimeError(
                    "downloaded streamed model has invalid expert artifacts: "
                    + str(expert_status["reason"])
                )
            raise RuntimeError(
                "downloaded model is incomplete: weight shards are missing or still partial"
            )
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "downloaded MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
        expert_status = expert_artifact_status(resolved)
        if expert_repo and not expert_status.get("streamed_experts"):
            raise RuntimeError(
                f"downloaded expert repository {repo_id} has no "
                f"{EXPERT_MANIFEST_FILE}"
            )
        if expert_status.get("streamed_experts"):
            admission = admit_expert_artifact(
                resolved,
                repo_id=repo_id,
                revision=inventory.resolved_revision,
                trusted_bank_digests=downloaded_digests,
            )
        _validate_pinned_laguna_files(resolved, repo_id)
        if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
            _write_source_marker(
                resolved,
                repo_id=repo_id,
                revision=revision,
            )
        final_size = directory_size_bytes(resolved)
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": final_size,
                "total_bytes": total_bytes if total_bytes else final_size,
                "delta_bytes": final_size - started_size,
            },
        )
    return {
        "repo_id": repo_id,
        "path": str(resolved),
        "cache_dir": str(root),
        "revision": revision,
        "resolved_revision": resolved_revision,
        "expert_admission": admission,
        "reused_existing": reused_existing,
        "resumed_existing": resumed_existing,
        "started_size_bytes": started_size,
        "size_bytes": directory_size_bytes(resolved),
        "has_runtime_contract": (resolved / "mtplx_runtime.json").exists(),
        "has_config": (resolved / "config.json").exists(),
        "validation": _pull_validation(resolved, repo_id),
    }


def remove_cached_model(model_ref: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref) or model_ref.replace("--", "/")
    path = cached_model_path(repo_id, cache_dir=cache_dir)
    existed = path.exists()
    size = directory_size_bytes(path) if existed else 0
    if existed:
        shutil.rmtree(path)
    return {
        "repo_id": repo_id,
        "path": str(path),
        "removed": existed,
        "size_bytes_removed": size,
    }


def hf_cache_report(*, cache_dir: str | Path | None = None) -> dict[str, Any]:
    root = model_cache_dir(cache_dir)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    token_source = "environment" if token_present else None
    if not token_present:
        try:
            from huggingface_hub import get_token

            token_present = bool(get_token())
            token_source = "huggingface_hub" if token_present else None
        except Exception:
            token_present = False
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        free_bytes: int | None = usage.free
    except OSError:
        free_bytes = None
    return {
        "cache_dir": str(root),
        "cache_exists": root.exists(),
        "cache_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "disk_free_bytes": free_bytes,
        "disk_free_gb": round(free_bytes / 1_000_000_000, 3) if free_bytes is not None else None,
        "cached_models": len(list_cached_models(cache_dir=root)),
        "token_present": token_present,
        "token_source": token_source,
    }
