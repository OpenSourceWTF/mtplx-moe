#!/usr/bin/env python3
"""Run provenance-rich deterministic AR benchmarks through streamed experts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
from contextlib import contextmanager
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifestError,
    load_expert_manifest,
    resolve_artifact_member,
)
from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes  # noqa: E402
from mtplx.benchmarks.resource_telemetry import (  # noqa: E402
    PowermetricsCollector,
    ResourceRun,
    ResourceTelemetrySampler,
)
from mtplx.runtime import load  # noqa: E402


_GLM52_AR_DEFAULTS = {
    "max_tokens": 65_536,
    "max_output_tokens": 131_072,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 0,
    "enable_thinking": True,
    "reasoning_effort": "max",
}
_HY3_AR_DEFAULTS = {
    "max_tokens": 65_536,
    "max_output_tokens": 262_144,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 0,
    "enable_thinking": False,
    "reasoning_effort": None,
}
_MODEL_DEFAULTS = {
    "glm52-q4": _GLM52_AR_DEFAULTS,
    "hy3-q4": _HY3_AR_DEFAULTS,
    "hy3-expert-only-q4": _HY3_AR_DEFAULTS,
    "hy3-expert-q2": _HY3_AR_DEFAULTS,
}


def model_defaults_for_key(model_key: str) -> dict[str, object]:
    """Return benchmark defaults without making local lanes auto-selectable."""

    return dict(_MODEL_DEFAULTS[model_key])


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, timeout=2
        ).strip()
    except Exception:
        return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _receipt_path(receipt_dir: Path, path: Path, *, suffix: str = "full") -> Path:
    key = _sha256_bytes(f"{path.resolve()}:{suffix}".encode("utf-8"))
    return receipt_dir.expanduser().resolve() / f"{key}.json"


def _verified_file_digest(
    path: Path,
    *,
    require_nocache: bool = False,
    receipt_path: Path | None = None,
) -> dict[str, object]:
    """Stream actual bytes, bypassing the page cache where the host supports it."""

    before = _file_stat_identity(path)
    if receipt_path is not None and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = None
        if (
            isinstance(receipt, dict)
            and receipt.get("schema") == "mtplx-verification-receipt-v1"
            and receipt.get("stat") == before
            and receipt.get("scope") == "full_file"
        ):
            return {
                "sha256": str(receipt["sha256"]),
                "size": before["size"],
                "verification_method": "versioned_prior_nocache_receipt",
                "page_cache_bypassed": True,
                "verification_elapsed_seconds": 0.0,
                "receipt_reused": True,
                "receipt": str(receipt_path),
            }
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    nocache = False
    started = time.perf_counter()
    try:
        command = getattr(fcntl, "F_NOCACHE", None)
        if command is not None:
            try:
                fcntl.fcntl(descriptor, command, 1)
                nocache = True
            except OSError:
                pass
        if require_nocache and not nocache:
            raise RuntimeError(
                f"non-caching verification is unavailable for retained evidence: {path}"
            )
        size = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024**2)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    elapsed = time.perf_counter() - started
    after = _file_stat_identity(path)
    if after != before:
        raise RuntimeError(f"artifact changed during verification: {path}")
    result = {
        "sha256": digest.hexdigest(),
        "size": size,
        "verification_method": "streamed_full_file_sha256",
        "page_cache_bypassed": nocache,
        "verification_elapsed_seconds": elapsed,
        "receipt_reused": False,
    }
    if receipt_path is not None and nocache:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "mtplx-verification-receipt-v1",
                        "scope": "full_file",
                        "stat": after,
                        "sha256": result["sha256"],
                    },
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, receipt_path)
        finally:
            temporary.unlink(missing_ok=True)
        result["receipt"] = str(receipt_path)
    return result


def _verified_range_digest(
    path: Path,
    offset: int,
    length: int,
    *,
    require_nocache: bool = False,
    receipt_path: Path | None = None,
) -> dict[str, object]:
    before = _file_stat_identity(path)
    if receipt_path is not None and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = None
        if (
            isinstance(receipt, dict)
            and receipt.get("schema") == "mtplx-verification-receipt-v1"
            and receipt.get("stat") == before
            and receipt.get("scope") == "byte_range"
            and receipt.get("offset") == offset
            and receipt.get("length") == length
        ):
            return {
                "sha256": str(receipt["sha256"]),
                "verification_method": "versioned_prior_nocache_receipt",
                "page_cache_bypassed": True,
                "verification_elapsed_seconds": 0.0,
                "receipt_reused": True,
                "receipt": str(receipt_path),
            }
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    nocache = False
    started = time.perf_counter()
    try:
        command = getattr(fcntl, "F_NOCACHE", None)
        if command is not None:
            try:
                fcntl.fcntl(descriptor, command, 1)
                nocache = True
            except OSError:
                pass
        if require_nocache and not nocache:
            raise RuntimeError(
                f"non-caching verification is unavailable for retained evidence: {path}"
            )
        remaining = length
        position = offset
        while remaining:
            chunk = os.pread(descriptor, min(8 * 1024**2, remaining), position)
            if not chunk:
                raise ValueError(f"resident range in {path.name} is truncated")
            digest.update(chunk)
            position += len(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    result = {
        "sha256": digest.hexdigest(),
        "verification_method": "streamed_resident_range_sha256",
        "page_cache_bypassed": nocache,
        "verification_elapsed_seconds": time.perf_counter() - started,
        "receipt_reused": False,
    }
    after = _file_stat_identity(path)
    if after != before:
        raise RuntimeError(f"artifact changed during verification: {path}")
    if receipt_path is not None and nocache:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "mtplx-verification-receipt-v1",
                        "scope": "byte_range",
                        "stat": after,
                        "offset": offset,
                        "length": length,
                        "sha256": result["sha256"],
                    },
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, receipt_path)
        finally:
            temporary.unlink(missing_ok=True)
        result["receipt"] = str(receipt_path)
    return result


def _verification_identity_fields(
    result: dict[str, object], *, report_timing: bool
) -> dict[str, object]:
    if report_timing:
        return result
    return {
        key: value
        for key, value in result.items()
        if key not in {"verification_elapsed_seconds", "receipt"}
    }


def build_expert_manifest_identity(path: Path) -> dict[str, object]:
    """Hash the small expert manifest and retain its pinned artifact metadata."""

    raw = path.expanduser().read_bytes()
    payload = json.loads(raw)
    return {
        "content_sha256": _sha256_bytes(raw),
        "declared_manifest_sha256": payload.get("manifest_sha256"),
        "model_key": payload.get("model_key"),
        "source_revision": payload.get("source_revision"),
        "source_repo": payload.get("source_repo"),
    }


def build_model_artifact_identity(
    model_root: Path,
    manifest_path: Path,
    *,
    require_nocache: bool = False,
    receipt_dir: Path | None = None,
    report_timing: bool = False,
) -> dict[str, object]:
    """Validate the manifest and identify the exact executable artifact bytes."""

    root = model_root.expanduser().resolve()
    raw_manifest = manifest_path.expanduser().read_bytes()
    manifest = load_expert_manifest(manifest_path, verify_digest=True)
    small_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    }
    small_names.update(path.name for path in root.glob("chat_template*.jinja"))
    small_files = []
    for name in sorted(small_names):
        path = resolve_artifact_member(root, name, must_exist=False)
        if path.is_file():
            verified = _verified_file_digest(
                path,
                require_nocache=require_nocache,
                receipt_path=(
                    _receipt_path(receipt_dir, path)
                    if receipt_dir is not None
                    else None
                ),
            )
            small_files.append(
                {
                    "name": name,
                    **_verification_identity_fields(
                        verified, report_timing=report_timing
                    ),
                }
            )

    if manifest.sidecar is not None:
        sidecar_path = resolve_artifact_member(root, manifest.sidecar.file)
        expert_payload = _verified_file_digest(
            sidecar_path,
            require_nocache=require_nocache,
            receipt_path=(
                _receipt_path(receipt_dir, sidecar_path)
                if receipt_dir is not None
                else None
            ),
        )
        if expert_payload["size"] != manifest.sidecar.size:
            raise ExpertManifestError("sidecar size differs from validated manifest")
        if expert_payload["sha256"] != manifest.sidecar.sha256:
            raise ExpertManifestError("sidecar digest differs from validated manifest")
        expert_payload.update(
            {
                "method": "verified_sidecar_sha256",
                "verification_level": "actual_full_file_digest_matches_manifest",
            }
        )
        expert_payload = _verification_identity_fields(
            expert_payload, report_timing=report_timing
        )
    else:
        expert_payload = {
            "method": "manifest_record_sha256_inventory",
            "verification_level": "per_record_declared_digest",
            "manifest_content_sha256": _sha256_bytes(raw_manifest),
        }

    resident_tensors = []
    seen_ranges: dict[str, list[tuple[int, int]]] = {}
    for tensor in manifest.resident_tensors:
        shard_path = resolve_artifact_member(root, tensor.shard)
        end = tensor.offset + tensor.length
        if end > shard_path.stat().st_size:
            raise ExpertManifestError(f"resident tensor {tensor.tensor} exceeds shard")
        shard_key = str(shard_path)
        ranges = seen_ranges.setdefault(shard_key, [])
        if any(
            tensor.offset < previous_end and previous_start < end
            for previous_start, previous_end in ranges
        ):
            raise ExpertManifestError("resident byte ranges overlap or repeat")
        ranges.append((tensor.offset, end))
        resident_tensors.append(
            {
                "tensor": tensor.tensor,
                "shard": tensor.shard,
                "offset": tensor.offset,
                "length": tensor.length,
                **_verification_identity_fields(
                    _verified_range_digest(
                        shard_path,
                        tensor.offset,
                        tensor.length,
                        require_nocache=require_nocache,
                        receipt_path=(
                            _receipt_path(
                                receipt_dir,
                                shard_path,
                                suffix=f"range-{tensor.offset}-{tensor.length}",
                            )
                            if receipt_dir is not None
                            else None
                        ),
                    ),
                    report_timing=report_timing,
                ),
            }
        )

    return {
        "method": "manifest_plus_executable_resident_content_v1",
        "verification_level": "full_small_files_and_authenticated_resident_content",
        "manifest": build_expert_manifest_identity(manifest_path),
        "expert_payload": expert_payload,
        "small_files": small_files,
        "resident_tensors": resident_tensors,
    }


def build_prompt_identity(prompt_text: str | None, prompt_ids) -> dict[str, object]:
    """Identify exact source content and the exact encoded token sequence."""

    content = (prompt_text or "").encode("utf-8")
    tokens = [int(token) for token in prompt_ids]
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return {
        "content_sha256": _sha256_bytes(content),
        "content_bytes": len(content),
        "token_sha256": _sha256_bytes(token_bytes),
        "token_count": len(tokens),
    }


def _safetensors_header_identity(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path.name} has no safetensors header length")
        header_size = int.from_bytes(header_size_raw, "little")
        if header_size <= 0 or header_size > 64 * 1024**2:
            raise ValueError(f"{path.name} has an invalid safetensors header length")
        header = handle.read(header_size)
    if len(header) != header_size:
        raise ValueError(f"{path.name} has a truncated safetensors header")
    return {
        "name": path.name,
        "size": size,
        "header_sha256": _sha256_bytes(header),
    }


def build_mtp_artifact_identity(
    artifact_root: Path,
    *,
    precision: str,
    require_nocache: bool = False,
    receipt_dir: Path | None = None,
    report_timing: bool = False,
) -> dict[str, object]:
    """Identify MTP artifacts from filenames, sizes, and small tensor headers."""

    filenames = (
        ("layer80-bf16.safetensors",)
        if precision == "bf16"
        else ("layer80-residents-q.safetensors", "layer80-q4.safetensors")
    )
    digest_manifest_path = artifact_root.expanduser() / "artifact-digests.json"
    declared_files = None
    if digest_manifest_path.is_file():
        digest_manifest = json.loads(digest_manifest_path.read_text(encoding="utf-8"))
        if digest_manifest.get("format") != "mtplx-artifact-digests-v1":
            raise ValueError("unsupported MTP artifact digest manifest format")
        declared_manifest_sha256 = digest_manifest.get("manifest_sha256")
        unsigned_manifest = {
            key: value
            for key, value in digest_manifest.items()
            if key != "manifest_sha256"
        }
        computed_manifest_sha256 = _sha256_bytes(
            json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if declared_manifest_sha256 != computed_manifest_sha256:
            raise ValueError("MTP artifact digest manifest authentication failed")
        declared_files = digest_manifest.get("files")
    files = []
    for filename in filenames:
        path = artifact_root.expanduser() / filename
        header_identity = _safetensors_header_identity(path)
        declared = declared_files.get(filename) if declared_files else None
        verified = _verified_file_digest(
            path,
            require_nocache=require_nocache,
            receipt_path=(
                _receipt_path(receipt_dir, path) if receipt_dir is not None else None
            ),
        )
        if declared is not None:
            if int(declared["size"]) != verified["size"]:
                raise ValueError(f"declared MTP size differs for {filename}")
            if str(declared["sha256"]) != verified["sha256"]:
                raise ValueError(f"declared MTP digest differs for {filename}")
            verification_level = "actual_full_file_digest_matches_manifest"
        else:
            verification_level = "full_file_verified_at_benchmark_start"
        files.append(
            {
                **header_identity,
                "full_sha256": verified["sha256"],
                "identity_method": verified["verification_method"],
                "page_cache_bypassed": verified["page_cache_bypassed"],
                "verification_level": verification_level,
                **(
                    {
                        "verification_elapsed_seconds": verified[
                            "verification_elapsed_seconds"
                        ],
                        "receipt_reused": verified["receipt_reused"],
                        "receipt": verified.get("receipt"),
                    }
                    if report_timing
                    else {}
                ),
            }
        )
    return {
        "precision": precision,
        "files": files,
    }


def build_harness_source_identity(root: Path = _ROOT) -> dict[str, object]:
    """Hash behavior-affecting local source independently of checkout path."""

    root = root.expanduser().resolve()
    candidates = set()
    for source_root in (
        root / "mtplx",
        root / "native_extensions",
        root / "vllm_metal",
    ):
        if not source_root.exists():
            continue
        for pattern in (
            "*.py",
            "*.metal",
            "*.c",
            "*.cc",
            "*.cpp",
            "*.h",
            "*.mm",
            "*.so",
            "*.dylib",
            "*.metallib",
            "CMakeLists.txt",
            "pyproject.toml",
        ):
            candidates.update(source_root.rglob(pattern))
    for relative in (
        "scripts/benchmark_streamed_generation.py",
        "scripts/analyze_expert_route_trace.py",
        "pyproject.toml",
        "uv.lock",
        "CMakeLists.txt",
    ):
        path = root / relative
        if path.is_file():
            candidates.add(path)
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(candidates)
    ]
    native_binaries = []
    extension_modules = {
        "mtplx_native_expert_io._ext",
        "mtplx_native_mlp._ext",
        "mlx.core",
    }
    extension_modules.update(
        name
        for name, module in sys.modules.items()
        if name.startswith(("mtplx", "mlx", "mtplx_native"))
        and getattr(module, "__file__", "")
        and Path(str(module.__file__)).suffix in {".so", ".dylib", ".metallib"}
    )
    seen_native_paths: set[Path] = set()
    for module_name in sorted(extension_modules):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError):
            spec = None
        origin = (
            Path(spec.origin).resolve() if spec is not None and spec.origin else None
        )
        related = []
        if origin is not None and origin.is_file():
            related = [origin, *origin.parent.glob("*.dylib")]
            related.extend(origin.parent.rglob("*.metallib"))
        for binary in related:
            binary = binary.resolve()
            if binary in seen_native_paths or binary.suffix not in {
                ".so",
                ".dylib",
                ".metallib",
            }:
                continue
            seen_native_paths.add(binary)
            native_binaries.append(
                {
                    "module": module_name,
                    "name": binary.name,
                    "sha256": _sha256_file(binary),
                    "size": binary.stat().st_size,
                }
            )
    dependency_versions = {}
    for distribution in ("mlx", "mlx-lm", "numpy", "safetensors"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    source_sha256 = _sha256_bytes(
        json.dumps(
            {
                "sources": inventory,
                "native_binaries": native_binaries,
                "dependency_versions": dependency_versions,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=2
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    *[item["path"] for item in inventory],
                ],
                cwd=root,
                text=True,
                timeout=5,
            ).strip()
        )
    except Exception:
        git_head = None
        dirty = None
    return {
        "method": "path_relative_source_inventory_sha256_v1",
        "source_sha256": source_sha256,
        "file_count": len(inventory),
        "native_binaries": native_binaries,
        "dependency_versions": dependency_versions,
        "git_head": git_head,
        "dirty": dirty,
    }


def settle_after_artifact_verification(
    identities: object,
    seconds: float,
    *,
    sleeper=time.sleep,
) -> bool:
    """Cooldown only after a fresh full/range verification, never receipt reuse."""

    if seconds < 0:
        raise ValueError("verification settle seconds must be non-negative")

    def fresh(value: object) -> bool:
        if isinstance(value, dict):
            if (value.get("verification_method") or value.get("identity_method")) in {
                "streamed_full_file_sha256",
                "streamed_resident_range_sha256",
            } and not value.get("receipt_reused", False):
                return True
            return any(fresh(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(fresh(item) for item in value)
        return False

    required = fresh(identities)
    if required and seconds:
        sleeper(seconds)
    return required


def stable_artifact_content_identity(value: object) -> object:
    """Remove run-local verification telemetry from performance fingerprints."""

    operational = {
        "verification_elapsed_seconds",
        "verification_method",
        "identity_method",
        "verification_level",
        "page_cache_bypassed",
        "receipt_reused",
        "receipt",
    }
    if isinstance(value, dict):
        return {
            key: stable_artifact_content_identity(item)
            for key, item in value.items()
            if key not in operational
        }
    if isinstance(value, list):
        return [stable_artifact_content_identity(item) for item in value]
    return value


def build_stable_performance_settings(
    *,
    runtime_config: Mapping[str, object],
    sampler: Mapping[str, object],
    seed: int,
    prompt_identity: Mapping[str, object],
    prompt_options: Mapping[str, object],
    generation: Mapping[str, object],
    scheduler: Mapping[str, object],
    mtp: Mapping[str, object],
    model_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Collect resolved, path-independent inputs used by config fingerprinting."""

    return {
        "runtime_config": dict(runtime_config),
        "sampler": dict(sampler),
        "seed": int(seed),
        "prompt_identity": dict(prompt_identity),
        "prompt_options": dict(prompt_options),
        "generation": dict(generation),
        "scheduler": dict(scheduler),
        "mtp": dict(mtp),
        "model_artifact": dict(model_artifact),
    }


def build_configuration_summary(
    base_run_label: str,
    *,
    cache_scope: str,
    slot_layout: str,
    concurrency: int,
    execution_lane: str,
    performance_settings: Mapping[str, object],
) -> dict[str, object]:
    """Return legacy run identity plus bounded requested-config identity."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    canonical_settings = {
        key: performance_settings[key] for key in sorted(performance_settings)
    }
    fingerprint_payload = {
        "cache_scope": cache_scope,
        "slot_layout": slot_layout,
        "requested_concurrency": concurrency,
        "execution_lane": execution_lane,
        "performance_settings": canonical_settings,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    configuration_label = (
        f"cache-{cache_scope}-layout-{slot_layout}-B{concurrency}"
        f"-lane-{execution_lane}-cfg-{fingerprint}"
    )
    return {
        "run_label": base_run_label,
        "configuration_label": configuration_label,
        "configuration_fingerprint": fingerprint,
        "cache_scope": cache_scope,
        "slot_layout": slot_layout,
        "concurrency": concurrency,
        "requested_concurrency": concurrency,
        "execution_lane": execution_lane,
        "performance_settings": canonical_settings,
    }


def build_response_filename(
    model_key: str,
    run_label: str,
    repeat: int,
    *,
    request_id: str | None = None,
    configuration_fingerprint: str | None = None,
) -> str:
    """Preserve the legacy filename, bounding only oversized path components."""

    request_suffix = f"-{request_id}" if request_id is not None else ""
    config_suffix = (
        f"-cfg-{configuration_fingerprint}"
        if configuration_fingerprint is not None
        else ""
    )
    tail = f"-repeat-{repeat}{request_suffix}{config_suffix}.md"
    candidate = f"{model_key}-{run_label}{tail}"
    if len(candidate.encode("utf-8")) <= 255:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    prefix = f"{model_key}-"
    hash_suffix = f"-h{digest}"
    available = 255 - len((prefix + hash_suffix + tail).encode("utf-8"))
    bounded_label = run_label.encode("utf-8")[:available].decode(
        "utf-8", errors="ignore"
    )
    return f"{prefix}{bounded_label}{hash_suffix}{tail}"


def write_response_file(
    output_dir: Path,
    text: str,
    *,
    model_key: str,
    run_label: str,
    repeat: int,
    configuration_fingerprint: str,
    request_id: str | None = None,
) -> Path:
    """Write once, falling back to config identity without overwriting evidence."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = (
        build_response_filename(
            model_key,
            run_label,
            repeat,
            request_id=request_id,
        ),
        build_response_filename(
            model_key,
            run_label,
            repeat,
            request_id=request_id,
            configuration_fingerprint=configuration_fingerprint,
        ),
    )
    for filename in candidates:
        path = output_dir / filename
        try:
            reservation = reserve_json_evidence_targets(path, None)
        except FileExistsError:
            continue
        try:
            reservation.commit(path, text)
        finally:
            reservation.cleanup()
        return path
    raise FileExistsError(
        f"response evidence already exists for run {run_label!r}, repeat {repeat}, "
        f"configuration {configuration_fingerprint}"
    )


class JsonEvidenceReservations:
    """Own sibling locks while publishing immutable JSON with no replacement."""

    def __init__(
        self,
        paths: tuple[Path, ...],
        locks: Mapping[Path, tuple[Path, str, int]],
    ) -> None:
        self.paths = paths
        self._locks = dict(locks)
        self._committed: set[Path] = set()

    def _assert_lock_owned(self, path: Path) -> None:
        lock_path, token, inode = self._locks[path]
        try:
            stat = lock_path.stat()
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"JSON evidence reservation was lost: {path}") from exc
        if stat.st_ino != inode or payload.get("owner_token") != token:
            raise RuntimeError(f"JSON evidence reservation ownership changed: {path}")

    def commit(self, path: Path, text: str) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in self.paths:
            raise ValueError("JSON evidence path was not reserved")
        if resolved in self._committed:
            raise FileExistsError(f"JSON evidence was already committed: {resolved}")
        self._assert_lock_owned(resolved)
        token = self._locks[resolved][1]
        temporary = resolved.with_name(f".{resolved.name}.tmp-{token}")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            self._assert_lock_owned(resolved)
            try:
                os.link(temporary, resolved)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"JSON evidence target appeared before publication: {resolved}"
                ) from exc
            directory_fd = os.open(resolved.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._committed.add(resolved)
        finally:
            temporary.unlink(missing_ok=True)

    def cleanup(self) -> None:
        for path in self.paths:
            lock_path, token, inode = self._locks[path]
            try:
                stat = lock_path.stat()
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                if stat.st_ino == inode and payload.get("owner_token") == token:
                    lock_path.unlink()
            except (FileNotFoundError, json.JSONDecodeError):
                pass


def reserve_json_evidence_targets(
    output_json: Path | None,
    route_trace_json: Path | None,
) -> JsonEvidenceReservations:
    """Exclusively reserve explicit JSON evidence paths before model loading."""

    paths = tuple(
        path.expanduser().resolve()
        for path in (output_json, route_trace_json)
        if path is not None
    )
    if len(paths) != len(set(paths)):
        raise ValueError("--output-json and --route-trace-json must be different")
    locks: dict[Path, tuple[Path, str, int]] = {}
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            lock_path = path.with_name(f".{path.name}.lock")
            token = secrets.token_hex(16)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                payload = json.dumps(
                    {"owner_token": token, "pid": os.getpid(), "target": path.name}
                ).encode("utf-8")
                os.write(descriptor, payload)
                os.fsync(descriptor)
                inode = os.fstat(descriptor).st_ino
            finally:
                os.close(descriptor)
            locks[path] = (lock_path, token, inode)
    except FileExistsError as exc:
        for lock_path, _token, _inode in locks.values():
            lock_path.unlink(missing_ok=True)
        raise FileExistsError(
            "JSON evidence target or ownership lock already exists; abandoned "
            f"locks require explicit operator removal: {exc.filename or exc}"
        ) from exc
    return JsonEvidenceReservations(paths, locks)


def resolve_run_label(args: argparse.Namespace) -> str:
    """Return the unchanged legacy user label or manifest-stem default."""

    return args.run_label or args.manifest.stem


def build_evidence_summary(
    rows: list[dict],
    *,
    configuration_label: str,
    requested_concurrency: int,
) -> dict[str, object]:
    """Summarize achieved concurrency without relabeling requested config."""

    achieved_peak_concurrency = max(
        (int(row["achieved_peak_concurrency"]) for row in rows),
        default=0,
    )
    undersubscribed = any(bool(row["undersubscribed"]) for row in rows)
    evidence_label = (
        f"{configuration_label}-achieved-B{achieved_peak_concurrency}"
        f"{'-undersubscribed' if undersubscribed else ''}"
    )
    streams = [stream for row in rows for stream in row.get("streams", [])]
    timing_summary = (
        summarize_stream_timings(
            streams,
            requested_concurrency=requested_concurrency,
            achieved_peak_concurrency=achieved_peak_concurrency,
            undersubscribed=undersubscribed,
        )
        if streams
        else None
    )
    return {
        "configuration_label": configuration_label,
        "evidence_label": evidence_label,
        "requested_concurrency": requested_concurrency,
        "achieved_peak_concurrency": achieved_peak_concurrency,
        "saturation_valid": bool(rows) and not undersubscribed,
        "undersubscribed": undersubscribed,
        "timing_summary": timing_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--model-key",
        choices=[
            "hy3-q4",
            "glm52-q4",
            "hy3-expert-only-q4",
            "hy3-expert-q2",
        ],
        required=True,
    )
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, required=True)
    parser.add_argument("--runtime-reserve", default="16GiB")
    parser.add_argument("--expert-cache-limit")
    parser.add_argument(
        "--cache-policy",
        choices=["frequency", "lru"],
        default="frequency",
    )
    parser.add_argument(
        "--cache-scope",
        choices=["layer", "global"],
        default="layer",
        help="Partition persistent expert records by layer or share them globally.",
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--prompt",
    )
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--context-tokens",
        type=_positive_int,
        help="Build an exact-size MTPLX prefill-ladder prompt for comparison runs.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["coding-agent", "legacy-repeat"],
        default="coding-agent",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode the prompt as a user turn with the artifact chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a precise senior software engineer. Give a complete, self-contained answer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--generation-profile",
        choices=["model-default", "deterministic", "qwen36-comparable"],
        default="deterministic",
        help=(
            "Sampling profile. Defaults to deterministic greedy so that "
            "unflagged runs are reproducible and comparable; pass "
            "model-default explicitly for vendor sampling."
        ),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--window-tokens",
        type=_positive_int,
        default=32,
        help="Capture rolling decode/cache telemetry every N generated tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=256,
        help=(
            "Maximum generated tokens; generation still stops naturally at EOS. "
            "Defaults to a bounded 256 so an unflagged run cannot decode for "
            "hours; pass the documented model ceiling explicitly for "
            "full-response lanes (65,536 for both profiles; GLM-5.2's hard "
            "output max is 131,072)."
        ),
    )
    parser.add_argument(
        "--window-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture a full streaming snapshot at each rolling window. "
            "Disable for headline runs: the snapshot walks every slot "
            "condition and contends with in-flight miss loads."
        ),
    )
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture diagnostic resource throughput, queue occupancy, and "
            "same-interval I/O/Metal coactivity. Off by default; enabled runs "
            "are diagnostic, not headline timing lanes."
        ),
    )
    parser.add_argument(
        "--resource-sample-interval",
        type=_positive_float,
        default=0.25,
        help="Seconds between cheap resource snapshots (default: 0.25).",
    )
    parser.add_argument(
        "--resource-max-samples",
        type=_positive_int,
        default=4096,
        help="Bounded resource timeline length (default: 4096 samples).",
    )
    parser.add_argument(
        "--ssd-ceiling-gib-s",
        type=_positive_float,
        help="Measured SSD ceiling used only for saturation evidence.",
    )
    parser.add_argument(
        "--powermetrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add non-interactive per-process CPU/GPU/wait/I/O samples. "
            "Requires --resource-telemetry and passwordless sudo authorization."
        ),
    )
    parser.add_argument("--repeats", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-between", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help=(
            "Saturation-lane stream count: decode N identical prompts through "
            "the streamed continuous-batch runner and report aggregate and "
            "per-stream tok/s. B=1 uses the same runner as B=2/4/8. "
            "Streams whose combined prompt+max-tokens KV exceeds "
            "--max-live-kv-tokens are serialized at step boundaries. "
            "Outputs at different concurrencies are not token-comparable: "
            "batch size is part of the run configuration label."
        ),
    )
    parser.add_argument(
        "--max-prefills-per-step",
        type=_positive_int,
        default=1,
        help=(
            "Joining prefills allowed per decode step boundary while other "
            "streams are actively decoding (concurrency > 1 only)."
        ),
    )
    parser.add_argument(
        "--workload-shape",
        choices=["static", "mixed-join"],
        default="static",
        help=(
            "Submit every stream before decode (static), or start one decoder "
            "and submit the remaining streams at --join-after-step so their "
            "bounded prefills run beside live decode (mixed-join)."
        ),
    )
    parser.add_argument(
        "--join-after-step",
        type=int,
        default=2,
        help="Decode step boundary that submits mixed-join requests (default: 2).",
    )
    parser.add_argument(
        "--reference-ar",
        action="store_true",
        help=(
            "Opt into the legacy single-stream generate_ar reference path. "
            "Unflagged AR runs, including B=1, use the continuous-batch runner "
            "so B=1/2/4/8 saturation evidence shares one execution path."
        ),
    )
    parser.add_argument(
        "--transient-slots",
        type=_positive_int,
        help="Global miss-service/I/O slots (default: model top-k).",
    )
    parser.add_argument(
        "--read-chunk",
        default="8MiB",
        help="Maximum native positional-read chunk (default: 8MiB).",
    )
    parser.add_argument(
        "--f-nocache",
        action="store_true",
        help="Use macOS F_NOCACHE reads directly into shared expert slots.",
    )
    parser.add_argument(
        "--verification-receipt-dir",
        type=Path,
        default=Path("~/.cache/mtplx/verification-receipts"),
        help="Versioned stat-bound receipts for prior full non-caching reads.",
    )
    parser.add_argument(
        "--verification-settle-seconds",
        type=float,
        default=5.0,
        help="Cooldown after fresh artifact digest reads and before timed generation.",
    )
    parser.add_argument(
        "--slot-layout",
        choices=["direct-slots", "component-banks", "metal-mmap"],
        default="direct-slots",
    )
    parser.add_argument(
        "--verified-sidecar",
        action="store_true",
        help="Verify the full sidecar once at open, then skip repeated record hashes.",
    )
    parser.add_argument(
        "--trust-sidecar",
        action="store_true",
        help=(
            "Explicitly trust the manifest's sidecar digest and skip both "
            "startup and per-record hashing. Intended for an unchanged local "
            "sidecar that was fully verified earlier."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save each generated response as Markdown in this directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Also write the complete benchmark payload to this JSON path.",
    )
    parser.add_argument(
        "--run-label",
        help="Filesystem-safe label used in saved response filenames.",
    )
    parser.add_argument(
        "--route-trace-json",
        type=Path,
        help="Save per-layer routed expert IDs for cache/prefetch simulation.",
    )
    parser.add_argument(
        "--enable-mtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Speculative decoding through the packaged layer-80 NextN head "
            "(hy3-q4 only; requires --mtp-artifacts). Default off: the AR "
            "path is unchanged unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--mtp-artifacts",
        type=Path,
        help=(
            "Directory holding the layer-80 Hy3 MTP head artifacts "
            "(layer80-bf16.safetensors for bf16; layer80-residents-q"
            ".safetensors and layer80-q4.safetensors for q4)."
        ),
    )
    parser.add_argument(
        "--mtp-precision",
        choices=("bf16", "q4"),
        help=(
            "Layer-80 NextN head precision (default bf16). bf16 loads the "
            "bit-exact BF16 head (~7.5 GB resident; quantized MTP heads "
            "collapse acceptance, docs/FORGE_BACKEND_CONTRACT.md section 6) "
            "- budget it against --expert-cache-limit. q4 loads the pinned "
            "quantized artifacts (~1.94 GiB expert bank). Requires "
            "--enable-mtp."
        ),
    )
    return parser


def validate_mtp_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.enable_mtp:
        if args.model_key != "hy3-q4":
            parser.error("--enable-mtp is packaged for --model-key hy3-q4 only")
        if args.mtp_artifacts is None:
            parser.error("--enable-mtp requires --mtp-artifacts")
        if getattr(args, "concurrency", 1) > 1:
            parser.error(
                "--enable-mtp is single-stream; the batch runner is AR-only "
                "and concurrent MTP requests would only queue"
            )
        if args.mtp_precision is None:
            args.mtp_precision = "bf16"
    elif args.mtp_artifacts is not None:
        parser.error("--mtp-artifacts requires --enable-mtp")
    elif args.mtp_precision is not None:
        parser.error("--mtp-precision requires --enable-mtp")


def validate_reference_ar_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.reference_ar and args.concurrency != 1:
        parser.error("--reference-ar requires --concurrency 1")
    if args.reference_ar and args.enable_mtp:
        parser.error("--reference-ar cannot be combined with --enable-mtp")


def validate_workload_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.join_after_step < 0:
        parser.error("--join-after-step must be non-negative")
    if args.workload_shape == "mixed-join" and args.concurrency < 2:
        parser.error("mixed-join requires --concurrency at least 2")
    if args.workload_shape == "mixed-join" and (args.reference_ar or args.enable_mtp):
        parser.error("mixed-join requires the continuous-batch AR lane")


def resolve_execution_lane(args: argparse.Namespace) -> str:
    """Return the explicit execution path used by this benchmark arm."""

    if args.enable_mtp:
        return "reference-mtp1"
    if args.reference_ar:
        return "reference-ar"
    return "continuous-batch-ar"


def validate_sidecar_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    manifest,
) -> None:
    """Fail closed when a sidecar trust mode has no validated sidecar target."""

    if args.verified_sidecar and args.trust_sidecar:
        parser.error("--verified-sidecar and --trust-sidecar are mutually exclusive")
    if (args.trust_sidecar or args.verified_sidecar) and manifest.sidecar is None:
        parser.error("sidecar trust/verification requires a validated manifest sidecar")


def should_verify_source_records(args: argparse.Namespace, manifest) -> bool:
    """Never let a no-sidecar trust flag disable source-record verification."""

    if args.slot_layout == "metal-mmap":
        return False
    return not (args.trust_sidecar and manifest.sidecar is not None)


def validate_resource_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.resource_max_samples < 2:
        parser.error("--resource-max-samples must be at least 2")
    if args.powermetrics and not args.resource_telemetry:
        parser.error("--powermetrics requires --resource-telemetry")
    if args.ssd_ceiling_gib_s is not None and not args.resource_telemetry:
        parser.error("--ssd-ceiling-gib-s requires --resource-telemetry")
    if args.ssd_ceiling_gib_s is not None and not args.f_nocache:
        parser.error("--ssd-ceiling-gib-s requires --f-nocache")


class _ConcurrentTokenCounter:
    def __init__(self) -> None:
        self._count = 0

    def observe(self, state, *, completed_tokens: int) -> None:
        observed = int(completed_tokens) + sum(
            int(stream.generated_tokens) for stream in state.live
        )
        self._count = max(self._count, observed)

    def finish(self, results) -> None:
        self._count = max(self._count, sum(len(result.tokens) for result in results))

    def count(self) -> int:
        return self._count


@contextmanager
def _resource_telemetry(args, runtime, token_count):
    if not args.resource_telemetry:
        yield None
        return
    sampler = ResourceTelemetrySampler(
        runtime.expert_resource_telemetry_snapshot,
        token_count=token_count,
        interval_s=args.resource_sample_interval,
        max_samples=args.resource_max_samples,
    )
    power = PowermetricsCollector(
        enabled=args.powermetrics,
        pid=os.getpid(),
        interval_ms=max(100, int(args.resource_sample_interval * 1000)),
    )
    with power, sampler:
        yield ResourceRun(sampler=sampler, powermetrics=power)


def _attach_resource_report(
    row: dict,
    resource_run: ResourceRun | None,
    *,
    ssd_ceiling_gib_s: float | None,
    generation_thread_cpu_ns: int,
    generation_elapsed_ns: int,
    final_completion_tokens: int,
) -> None:
    if resource_run is None:
        return
    row["diagnostic_run"] = True
    row["resource_telemetry"] = resource_run.report(
        ssd_ceiling_gib_s=ssd_ceiling_gib_s,
        generation_thread_cpu_ns=generation_thread_cpu_ns,
        generation_elapsed_ns=generation_elapsed_ns,
        final_completion_tokens=final_completion_tokens,
    )


def _run_reference_generation(
    args,
    runtime,
    *,
    prompt_ids,
    max_tokens: int,
    sampler,
    token_callback,
    resource_run: ResourceRun | None,
    generate_ar_fn,
    generate_mtp1_fn,
):
    thread_cpu_started = time.thread_time_ns() if resource_run is not None else 0
    started = time.perf_counter()
    with runtime.admit_kv_tokens(len(prompt_ids) + max_tokens):
        if args.enable_mtp:
            result = generate_mtp1_fn(
                runtime,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                seed=args.seed,
            )
        else:
            result = generate_ar_fn(
                runtime,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                seed=args.seed,
                token_callback=token_callback,
            )
    finished = time.perf_counter()
    thread_cpu_finished = time.thread_time_ns() if resource_run is not None else 0
    return result, started, finished, thread_cpu_started, thread_cpu_finished


def build_concurrent_requests(
    prompt_ids,
    *,
    concurrency: int,
    max_tokens: int,
    sampler,
    seed: int,
):
    """Build the saturation lane's N identical prompts as batch requests.

    Prompts are identical across streams; per-stream seeds are ``seed + i``
    so sampled profiles produce distinct streams while the deterministic
    profile stays seed-independent.
    """

    from mtplx.streamed_batch import StreamedBatchRequest

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    return [
        StreamedBatchRequest(
            request_id=f"stream-{index:02d}",
            prompt_ids=tuple(int(token) for token in prompt_ids),
            max_tokens=max_tokens,
            sampler=sampler,
            seed=seed + index,
        )
        for index in range(concurrency)
    ]


def _r7_percentile(values: list[float], percentile: int) -> float:
    """Return the deterministic R7 linearly interpolated percentile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentiles require at least one value")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_stream_timings(
    streams: list[dict],
    *,
    requested_concurrency: int,
    achieved_peak_concurrency: int,
    undersubscribed: bool | None = None,
) -> dict[str, object]:
    """Summarize comparable stream timings and saturation state."""

    ttft = [float(stream["ttft_seconds"]) for stream in streams]
    completion = [float(stream["completion_latency_seconds"]) for stream in streams]

    def percentiles(values: list[float]) -> dict[str, float]:
        return {
            "p50": _r7_percentile(values, 50),
            "p95": _r7_percentile(values, 95),
            "p99": _r7_percentile(values, 99),
        }

    if undersubscribed is None:
        undersubscribed = achieved_peak_concurrency < requested_concurrency
    return {
        "stream_count": len(streams),
        "percentile_method": "linear_interpolation_r7",
        "requested_concurrency": requested_concurrency,
        "achieved_peak_concurrency": achieved_peak_concurrency,
        "saturation_valid": not undersubscribed,
        "undersubscribed": undersubscribed,
        "ttft_seconds": percentiles(ttft),
        "completion_latency_seconds": percentiles(completion),
    }


def _run_concurrent_repeats(
    args,
    runtime,
    *,
    prompt_ids,
    sampler,
    max_tokens: int,
    run_label: str,
    configuration_label: str,
    configuration_fingerprint: str = "unfingerprinted",
) -> list[dict]:
    """Saturation lane: N identical prompts, aggregate and per-stream tok/s."""

    from mtplx.streamed_batch import StreamedBatchRunner

    rows: list[dict] = []
    for repeat in range(args.repeats):
        if args.reset_between and repeat:
            runtime.expert_streaming.reset()
        requests = build_concurrent_requests(
            prompt_ids,
            concurrency=args.concurrency,
            max_tokens=max_tokens,
            sampler=sampler,
            seed=args.seed,
        )
        before = runtime.expert_streaming_snapshot()
        workload_shape = getattr(args, "workload_shape", "static")
        join_after_step = int(getattr(args, "join_after_step", 2))
        telemetry_enabled = bool(args.resource_telemetry)
        runner_box: list[StreamedBatchRunner] = []
        join_submission_step: int | None = None
        token_counter = _ConcurrentTokenCounter() if telemetry_enabled else None

        def submit_joiners(state) -> None:
            nonlocal join_submission_step
            if (
                workload_shape == "mixed-join"
                and join_submission_step is None
                and state.step == join_after_step
            ):
                for request in requests[1:]:
                    runner_box[0].submit(request)
                join_submission_step = state.step

        def observe_step(state) -> None:
            assert token_counter is not None
            completed_tokens = sum(
                len(result.tokens) for result in runner_box[0]._results.values()
            )
            token_counter.observe(state, completed_tokens=completed_tokens)
            submit_joiners(state)

        if telemetry_enabled:
            step_callback = observe_step
        elif workload_shape == "mixed-join":
            step_callback = submit_joiners
        else:
            step_callback = None
        runner = StreamedBatchRunner(
            runtime,
            max_concurrency=args.concurrency,
            max_prefills_per_step=args.max_prefills_per_step,
            on_step=step_callback,
        )
        runner_box.append(runner)
        initial_requests = requests[:1] if workload_shape == "mixed-join" else requests
        for request in initial_requests:
            runner.submit(request)
        with _resource_telemetry(
            args,
            runtime,
            token_counter.count if token_counter is not None else None,
        ) as resource_run:
            if resource_run is None:
                thread_cpu_started = 0
                started = time.perf_counter()
                results = runner.run()
                if workload_shape == "mixed-join" and join_submission_step is None:
                    raise RuntimeError(
                        "initial stream finished before the mixed-join submission step"
                    )
                finished = time.perf_counter()
                thread_cpu_finished = 0
            else:
                thread_cpu_started = time.thread_time_ns()
                started = time.perf_counter()
                results = runner.run()
                if workload_shape == "mixed-join" and join_submission_step is None:
                    raise RuntimeError(
                        "initial stream finished before the mixed-join submission step"
                    )
                finished = time.perf_counter()
                thread_cpu_finished = time.thread_time_ns()
                assert token_counter is not None
                token_counter.finish(results)
        elapsed = finished - started
        after = runtime.expert_streaming_snapshot()
        streams = []
        for result in results:
            completion_tokens = len(result.tokens)
            stream_elapsed = result.last_token_s - result.admitted_s
            decode_elapsed = result.last_token_s - result.first_token_s
            ttft_seconds = result.first_token_s - result.admitted_s
            response_path = None
            if args.output_dir is not None:
                response_path = write_response_file(
                    args.output_dir,
                    result.text + "\n",
                    model_key=args.model_key,
                    run_label=run_label,
                    repeat=repeat,
                    configuration_fingerprint=configuration_fingerprint,
                    request_id=result.request_id,
                )
            streams.append(
                {
                    "request_id": result.request_id,
                    "seed": args.seed + len(streams),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_tokens_per_second": (
                        completion_tokens / stream_elapsed
                        if stream_elapsed > 0.0
                        else 0.0
                    ),
                    "decode_tokens_per_second": (
                        (completion_tokens - 1) / decode_elapsed
                        if completion_tokens > 1 and decode_elapsed > 0.0
                        else 0.0
                    ),
                    "finish_reason": result.finish_reason,
                    "admitted_step": result.admitted_step,
                    "finished_step": result.finished_step,
                    "decode_steps": result.decode_steps,
                    "prefill_seconds": result.prefill_seconds,
                    "ttft_seconds": ttft_seconds,
                    "completion_latency_seconds": stream_elapsed,
                    "token_times_s": [
                        token_time - started for token_time in result.token_times_s
                    ],
                    "token_ids": list(result.tokens),
                    "text": result.text,
                    "response_path": (
                        str(response_path) if response_path is not None else None
                    ),
                }
            )
        aggregate_tokens = sum(stream["completion_tokens"] for stream in streams)
        scheduler = runner.stats()
        achieved_peak_concurrency = max(scheduler["live_stream_counts"], default=0)
        undersubscribed = achieved_peak_concurrency < args.concurrency
        timing_summary = summarize_stream_timings(
            streams,
            requested_concurrency=args.concurrency,
            achieved_peak_concurrency=achieved_peak_concurrency,
            undersubscribed=undersubscribed,
        )
        evidence_label = (
            f"{configuration_label}-achieved-B{achieved_peak_concurrency}"
            f"{'-undersubscribed' if undersubscribed else ''}"
        )
        row = {
            "repeat": repeat,
            "elapsed_seconds": elapsed,
            "prompt_tokens": len(prompt_ids),
            "run_label": run_label,
            "configuration_label": configuration_label,
            "concurrency": args.concurrency,
            "requested_concurrency": args.concurrency,
            "achieved_peak_concurrency": achieved_peak_concurrency,
            "saturation_valid": not undersubscribed,
            "undersubscribed": undersubscribed,
            "evidence_label": evidence_label,
            "cache_scope": args.cache_scope,
            "slot_layout": args.slot_layout,
            "execution_lane": "continuous-batch-ar",
            "workload_shape": workload_shape,
            "join_submission_step": join_submission_step,
            "aggregate_completion_tokens": aggregate_tokens,
            "aggregate_completion_tokens_per_second": (
                aggregate_tokens / elapsed if elapsed > 0.0 else 0.0
            ),
            "scheduler": scheduler,
            "timing_summary": timing_summary,
            "streams": streams,
            "streaming_before": before,
            "streaming_after": after,
        }
        _attach_resource_report(
            row,
            resource_run,
            ssd_ceiling_gib_s=args.ssd_ceiling_gib_s,
            generation_thread_cpu_ns=thread_cpu_finished - thread_cpu_started,
            generation_elapsed_ns=int(elapsed * 1e9),
            final_completion_tokens=aggregate_tokens,
        )
        rows.append(row)
    return rows


_active_evidence_reservations: JsonEvidenceReservations | None = None


def _main() -> int:
    global _active_evidence_reservations
    parser = build_parser()
    args = parser.parse_args()
    validate_resource_flags(parser, args)
    root = args.model_root.expanduser().resolve()
    model_defaults = model_defaults_for_key(args.model_key)
    if args.generation_profile == "deterministic":
        profile_defaults = {
            **model_defaults,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "enable_thinking": False,
            "reasoning_effort": None,
        }
    elif args.generation_profile == "qwen36-comparable":
        profile_defaults = {
            **model_defaults,
            "max_tokens": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "enable_thinking": False,
            "reasoning_effort": None,
        }
    else:
        profile_defaults = model_defaults
    max_tokens = args.max_tokens or int(profile_defaults["max_tokens"])
    benchmark_context_limit = 262_144
    if args.max_live_kv_tokens > benchmark_context_limit:
        parser.error(
            f"--max-live-kv-tokens {args.max_live_kv_tokens} exceeds the current "
            f"benchmark ceiling of {benchmark_context_limit}"
        )
    if max_tokens > int(model_defaults["max_output_tokens"]):
        parser.error(
            f"--max-tokens {max_tokens} exceeds {args.model_key}'s documented "
            f"maximum output of {model_defaults['max_output_tokens']}"
        )
    if args.verification_settle_seconds < 0:
        parser.error("--verification-settle-seconds must be non-negative")
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(profile_defaults["temperature"])
    )
    top_p = args.top_p if args.top_p is not None else float(profile_defaults["top_p"])
    top_k = args.top_k if args.top_k is not None else int(profile_defaults["top_k"])
    enable_thinking = (
        args.enable_thinking
        if args.enable_thinking is not None
        else bool(profile_defaults["enable_thinking"])
    )
    reasoning_effort = args.reasoning_effort or profile_defaults["reasoning_effort"]
    base_run_label = resolve_run_label(args)
    if not base_run_label or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in base_run_label
    ):
        parser.error("--run-label may contain only letters, digits, '-' and '_'")
    run_label = base_run_label
    validate_mtp_flags(parser, args)
    validate_reference_ar_flags(parser, args)
    validate_workload_flags(parser, args)
    execution_lane = resolve_execution_lane(args)
    evidence_reservations = reserve_json_evidence_targets(
        args.output_json, args.route_trace_json
    )
    _active_evidence_reservations = evidence_reservations
    validated_manifest = (
        load_expert_manifest(args.manifest, verify_digest=True)
        if args.trust_sidecar or args.verified_sidecar
        else None
    )
    if validated_manifest is not None:
        validate_sidecar_flags(parser, args, validated_manifest)
    config = ExpertStreamingConfig(
        model_key=args.model_key,
        memory_limit_bytes=parse_memory_bytes(args.memory_limit),
        max_live_kv_tokens=args.max_live_kv_tokens,
        runtime_reserve_bytes=parse_memory_bytes(args.runtime_reserve),
        expert_cache_limit_bytes=(
            parse_memory_bytes(args.expert_cache_limit)
            if args.expert_cache_limit
            else None
        ),
        cache_policy=args.cache_policy,
        cache_scope=args.cache_scope,
        transient_slots=args.transient_slots,
        max_read_chunk_bytes=parse_memory_bytes(args.read_chunk),
        bypass_page_cache=args.f_nocache,
        slot_layout=args.slot_layout,
        verify_record_hashes=should_verify_source_records(args, validated_manifest),
        verify_sidecar_hash_at_open=args.verified_sidecar,
        trace_routes=args.route_trace_json is not None,
        resource_telemetry=args.resource_telemetry,
    )
    verification_receipt_dir = args.verification_receipt_dir.expanduser().resolve()
    model_artifact_identity = build_model_artifact_identity(
        root,
        args.manifest,
        require_nocache=True,
        receipt_dir=verification_receipt_dir,
        report_timing=True,
    )
    model_artifact_identity["harness_source"] = build_harness_source_identity()
    mtp_artifact_identity = (
        build_mtp_artifact_identity(
            args.mtp_artifacts,
            precision=(args.mtp_precision or "bf16"),
            require_nocache=True,
            receipt_dir=verification_receipt_dir,
            report_timing=True,
        )
        if args.enable_mtp and args.mtp_artifacts is not None
        else None
    )
    verification_settle_applied = settle_after_artifact_verification(
        (model_artifact_identity, mtp_artifact_identity),
        args.verification_settle_seconds,
    )
    try:
        runtime = load(
            root,
            mtp=args.enable_mtp,
            expert_streaming_config=config,
            expert_manifest=args.manifest,
            mtp_artifacts=(
                args.mtp_artifacts.expanduser().resolve()
                if args.mtp_artifacts is not None
                else None
            ),
            mtp_precision=(args.mtp_precision or "bf16"),
        )
    except BaseException:
        evidence_reservations.cleanup()
        raise
    rows = []
    try:
        from mtplx.generation import generate_ar, generate_mtp1
        from mtplx.sampling import SamplerConfig

        prompt_text = (
            args.prompt_file.expanduser().read_text(encoding="utf-8")
            if args.prompt_file is not None
            else args.prompt
        )
        default_prompt = "Explain why the sky is blue in one paragraph."
        effective_prompt_text = (
            prompt_text
            if args.context_tokens is not None
            else (prompt_text or default_prompt)
        )
        prompt_metadata = None
        if args.context_tokens is not None:
            from mtplx.prefill_bench import _prompt_build_for_context

            prompt_build = _prompt_build_for_context(
                runtime.tokenizer,
                args.context_tokens,
                prompt_style=args.prompt_style,
                prompt_tail=prompt_text,
                prompt_format="chat" if args.chat else "raw",
                enable_thinking=enable_thinking,
            )
            prompt_ids = prompt_build.token_ids
            prompt_metadata = prompt_build.metadata
        elif args.chat:
            from mtplx.chat_encoding import encode_chat_messages

            prompt_ids = encode_chat_messages(
                runtime.tokenizer,
                [
                    {"role": "system", "content": args.system_prompt},
                    {
                        "role": "user",
                        "content": effective_prompt_text,
                    },
                ],
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        else:
            prompt_ids = runtime.tokenizer.encode(effective_prompt_text)
        sampler = SamplerConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        performance_settings = build_stable_performance_settings(
            runtime_config=config.to_dict(),
            sampler={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            seed=args.seed,
            prompt_identity=build_prompt_identity(effective_prompt_text, prompt_ids),
            prompt_options={
                "chat": args.chat,
                "system_prompt": args.system_prompt,
                "prompt_style": args.prompt_style,
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
                "prompt_metadata": prompt_metadata,
            },
            generation={
                "generation_profile": args.generation_profile,
                "max_tokens": max_tokens,
                "context_tokens": args.context_tokens,
                "window_tokens": args.window_tokens,
                "window_telemetry": args.window_telemetry,
                "repeats": args.repeats,
                "reset_between": args.reset_between,
            },
            scheduler={
                "requested_concurrency": args.concurrency,
                "max_prefills_per_step": args.max_prefills_per_step,
                "execution_lane": execution_lane,
                "workload_shape": args.workload_shape,
                "join_after_step": (
                    args.join_after_step
                    if args.workload_shape == "mixed-join"
                    else None
                ),
            },
            mtp={
                "enabled": args.enable_mtp,
                "precision": args.mtp_precision or "bf16",
                "artifact_identity": stable_artifact_content_identity(
                    mtp_artifact_identity
                ),
            },
            model_artifact=stable_artifact_content_identity(model_artifact_identity),
        )
        configuration_summary = build_configuration_summary(
            base_run_label,
            cache_scope=args.cache_scope,
            slot_layout=args.slot_layout,
            concurrency=args.concurrency,
            execution_lane=execution_lane,
            performance_settings=performance_settings,
        )
        configuration_label = str(configuration_summary["configuration_label"])
        configuration_fingerprint = str(
            configuration_summary["configuration_fingerprint"]
        )
        if execution_lane == "continuous-batch-ar":
            rows.extend(
                _run_concurrent_repeats(
                    args,
                    runtime,
                    prompt_ids=prompt_ids,
                    sampler=sampler,
                    max_tokens=max_tokens,
                    run_label=run_label,
                    configuration_label=configuration_label,
                    configuration_fingerprint=configuration_fingerprint,
                )
            )
        # Legacy AR and MTP generation remain explicitly labelled references.
        for repeat in range(
            args.repeats if execution_lane != "continuous-batch-ar" else 0
        ):
            if args.reset_between and repeat:
                runtime.expert_streaming.reset()
            before = runtime.expert_streaming_snapshot()
            decode_points = []
            decoded_count = 0

            def token_callback(token_ids):
                nonlocal decoded_count
                decoded_count += len(token_ids)
                if decoded_count == 1 or (decoded_count - 1) % args.window_tokens == 0:
                    decode_points.append(
                        {
                            "completion_tokens": decoded_count,
                            "time": time.perf_counter(),
                            "streaming": (
                                runtime.expert_streaming_snapshot()
                                if args.window_telemetry
                                else None
                            ),
                        }
                    )

            with _resource_telemetry(
                args,
                runtime,
                lambda: decoded_count,
            ) as resource_run:
                (
                    result,
                    started,
                    finished,
                    thread_cpu_started,
                    thread_cpu_finished,
                ) = _run_reference_generation(
                    args,
                    runtime,
                    prompt_ids=prompt_ids,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    token_callback=token_callback,
                    resource_run=resource_run,
                    generate_ar_fn=generate_ar,
                    generate_mtp1_fn=generate_mtp1,
                )
                decoded_count = len(result.tokens)
            elapsed = finished - started
            after = runtime.expert_streaming_snapshot()
            token_ids = [int(token) for token in result.tokens]
            if token_ids and (
                not decode_points
                or decode_points[-1]["completion_tokens"] != len(token_ids)
            ):
                # Stamp the final window with the generation end time, not a
                # timestamp taken after the full-slot snapshot walk above.
                decode_points.append(
                    {
                        "completion_tokens": len(token_ids),
                        "time": finished,
                        "streaming": after,
                    }
                )
            rolling_decode = []
            for left, right in zip(decode_points, decode_points[1:], strict=False):
                window_elapsed = right["time"] - left["time"]
                window_tokens = right["completion_tokens"] - left["completion_tokens"]
                rolling_decode.append(
                    {
                        "from_completion_token": left["completion_tokens"],
                        "to_completion_token": right["completion_tokens"],
                        "decode_tokens": window_tokens,
                        "elapsed_seconds": window_elapsed,
                        "decode_tokens_per_second": (
                            window_tokens / window_elapsed
                            if window_elapsed > 0.0
                            else 0.0
                        ),
                        "streaming_before": left["streaming"],
                        "streaming_after": right["streaming"],
                    }
                )
            response_path = None
            if args.output_dir is not None:
                response_path = write_response_file(
                    args.output_dir,
                    result.text + "\n",
                    model_key=args.model_key,
                    run_label=run_label,
                    repeat=repeat,
                    configuration_fingerprint=configuration_fingerprint,
                )
            row = {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "prompt_tokens": len(prompt_ids),
                "run_label": run_label,
                "configuration_label": configuration_label,
                "concurrency": args.concurrency,
                "requested_concurrency": args.concurrency,
                "achieved_peak_concurrency": 1,
                "saturation_valid": args.concurrency == 1,
                "undersubscribed": args.concurrency > 1,
                "evidence_label": f"{configuration_label}-achieved-B1",
                "cache_scope": args.cache_scope,
                "slot_layout": args.slot_layout,
                "execution_lane": execution_lane,
                "completion_tokens": len(token_ids),
                "completion_tokens_per_second": len(token_ids) / elapsed,
                "token_ids": token_ids,
                "text": result.text,
                "response_path": (
                    str(response_path) if response_path is not None else None
                ),
                "finish_reason": result.finish_reason,
                "rolling_decode": rolling_decode,
                "streaming_before": before,
                "streaming_after": after,
                "generation_stats": result.stats.to_dict(),
            }
            _attach_resource_report(
                row,
                resource_run,
                ssd_ceiling_gib_s=args.ssd_ceiling_gib_s,
                generation_thread_cpu_ns=thread_cpu_finished - thread_cpu_started,
                generation_elapsed_ns=int(elapsed * 1e9),
                final_completion_tokens=len(token_ids),
            )
            rows.append(row)
    except BaseException:
        evidence_reservations.cleanup()
        raise
    finally:
        runtime.close(timeout=10.0)

    evidence_summary = build_evidence_summary(
        rows,
        configuration_label=configuration_label,
        requested_concurrency=args.concurrency,
    )
    payload = {
        "schema": "mtplx-streamed-generation-benchmark-v1",
        "git_commit": _git_commit(),
        "model_root": str(root),
        "model_key": args.model_key,
        "manifest": str(args.manifest.resolve()),
        "config": config.to_dict(),
        "seed": args.seed,
        "chat": args.chat,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "mtp": {
            "enabled": args.enable_mtp,
            "artifacts": (
                str(args.mtp_artifacts.expanduser().resolve())
                if args.mtp_artifacts is not None
                else None
            ),
            "precision": args.mtp_precision,
        },
        "generation_profile": args.generation_profile,
        "run_label": run_label,
        "configuration_label": configuration_label,
        "cache_scope": args.cache_scope,
        "slot_layout": args.slot_layout,
        "execution_lane": execution_lane,
        "workload_shape": args.workload_shape,
        "join_after_step": (
            args.join_after_step if args.workload_shape == "mixed-join" else None
        ),
        "concurrency": args.concurrency,
        "requested_concurrency": args.concurrency,
        "achieved_peak_concurrency": evidence_summary["achieved_peak_concurrency"],
        "saturation_valid": evidence_summary["saturation_valid"],
        "undersubscribed": evidence_summary["undersubscribed"],
        "configuration_summary": configuration_summary,
        "artifact_verification": {
            "model": model_artifact_identity,
            "mtp": mtp_artifact_identity,
            "receipt_dir": str(verification_receipt_dir),
            "settle_seconds": args.verification_settle_seconds,
            "settle_applied": verification_settle_applied,
            "timing_scope": "outside_timed_generation",
        },
        "evidence_summary": evidence_summary,
        "max_prefills_per_step": args.max_prefills_per_step,
        "generation": {
            "max_tokens": max_tokens,
            "documented_max_output_tokens": model_defaults["max_output_tokens"],
            "benchmark_context_limit": benchmark_context_limit,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "prompt_file": (
            str(args.prompt_file.expanduser().resolve())
            if args.prompt_file is not None
            else None
        ),
        "context_tokens": args.context_tokens,
        "prompt_style": args.prompt_style if args.context_tokens is not None else None,
        "prompt_metadata": prompt_metadata,
        "reset_between": args.reset_between,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": rows,
    }
    if args.route_trace_json is not None:
        route_trace_json = args.route_trace_json.expanduser().resolve()
        route_trace_json.parent.mkdir(parents=True, exist_ok=True)
        route_trace_payload = {
            "schema": "mtplx-expert-route-trace-v2",
            "model_key": args.model_key,
            "manifest_sha256": runtime.expert_streaming.manifest.manifest_sha256,
            "reset_behavior": (
                "trace_epoch increments and decode_step restarts at zero after reset"
            ),
            "repeats": args.repeats,
            "reset_between": args.reset_between,
            "transient_slots": runtime.expert_streaming.plan.transient_slots,
            "entries": runtime.expert_streaming.route_trace(),
        }
        evidence_reservations.commit(
            route_trace_json,
            json.dumps(route_trace_payload, indent=2, sort_keys=True) + "\n",
        )
        payload["route_trace_json"] = str(route_trace_json)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        evidence_reservations.commit(output_json, rendered + "\n")
    print(rendered)
    return 0


def main() -> int:
    """Run the CLI with explicit cleanup spanning every post-reservation phase."""

    global _active_evidence_reservations
    try:
        return _main()
    finally:
        if _active_evidence_reservations is not None:
            _active_evidence_reservations.cleanup()
            _active_evidence_reservations = None


if __name__ == "__main__":
    raise SystemExit(main())
