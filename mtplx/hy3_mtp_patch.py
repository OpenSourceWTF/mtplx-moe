"""Runtime MTP injection for streamed Hy3: the layer-80 NextN head.

The pinned pipenetwork/Hy3-4bit artifact omits checkpoint layer 80, so the
head is packaged separately from the official tencent/Hy3 BF16 weights:

    layer80-bf16.safetensors         the whole head bit-exact from the source
                                     checkpoint: every layer-80 tensor BF16
                                     (expert_bias F32), one .weight leaf per
                                     expert projection
                                     (scripts/extract_mtp_layer80.py)
    layer80-residents-q.safetensors  attention/router/shared/norms/eh_proj in
                                     the pinned resident conventions
                                     (scripts/quantize_mtp_layer80_residents.py)
    layer80-q4.safetensors           192 routed experts in the pinned affine
                                     Q4/gs64 expert segment format
                                     (scripts/quantize_mtp_layer80.py)

The default precision is ``bf16``: docs/FORGE_BACKEND_CONTRACT.md section 6
documents that quantizing MTP weights collapses MoE acceptance to 5-11%
(vs 79-85% with the BF16 head).  ``q4`` keeps the quantized artifacts
selectable for memory-constrained A/B runs.  Memory: the BF16 head is
~7.5 GB (7.0 GiB) resident vs ~1.94 GiB for the Q4 expert bank — callers
must budget the difference in their expert-cache plans.

Unlike trunk layers 1-79 the head's experts are fully resident: they are
stacked into one ``SwitchGLU`` at MTP-enable time (quantized only in q4
mode).  Loading is fail-closed: unexpected, missing, or wrongly typed
tensors and revision mismatches abort instead of degrading.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator, Mapping

logger = logging.getLogger(__name__)

HY3_MTP_SOURCE_REPO = "tencent/Hy3"
HY3_MTP_SOURCE_REVISION = "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
HY3_MTP_BF16_FILE = "layer80-bf16.safetensors"
HY3_MTP_RESIDENTS_FILE = "layer80-residents-q.safetensors"
HY3_MTP_EXPERTS_FILE = "layer80-q4.safetensors"
HY3_MTP_PRECISIONS = ("bf16", "q4")
# Forge contract section 6: quantized MTP heads collapse acceptance, so the
# bit-exact BF16 head is the default despite its larger resident footprint.
HY3_MTP_DEFAULT_PRECISION = "bf16"

_QUANTIZED_RESIDENT_BASES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.shared_mlp.gate_proj",
    "mlp.shared_mlp.up_proj",
    "mlp.shared_mlp.down_proj",
    "mlp.router.gate",
)
_BF16_RESIDENT_SUFFIXES = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "final_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_F32_RESIDENT_SUFFIXES = ("mlp.expert_bias",)
_HEAD_LOCAL_MODULES = ("enorm", "hnorm", "eh_proj", "final_layernorm")
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_QUANT_LEAVES = ("weight", "scales", "biases")
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_MAX_SAFETENSORS_TENSORS = 4096
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class Hy3MTPLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedHy3MTPArtifacts:
    """Structurally verified artifacts held open for an exact runtime load."""

    root: Path
    precision: str
    payload_bytes: int
    source_revision: str
    files: Mapping[str, BinaryIO]


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _pread_exact(fd: int, size: int, offset: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        chunk = os.pread(fd, size - received, offset + received)
        if not chunk:
            raise Hy3MTPLoadError(f"{label} is truncated")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _tensor_payload_bytes(
    dtype: Any,
    shape: Any,
    *,
    path: Path,
    name: str,
) -> int:
    if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
        raise Hy3MTPLoadError(f"{path.name} tensor {name!r} has invalid dtype")
    if not isinstance(shape, list) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise Hy3MTPLoadError(f"{path.name} tensor {name!r} has invalid shape")
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * _SAFETENSORS_DTYPE_BYTES[dtype]


def _preflight_safetensors_payload_held(
    path: Path,
    fd: int,
    opened_info: os.stat_result,
    expected_revision: str,
) -> int:
    try:
        current_info = os.fstat(fd)
        if not stat.S_ISREG(current_info.st_mode) or not _unchanged_while_held(
            opened_info, current_info
        ):
            raise Hy3MTPLoadError(f"Hy3 MTP artifact changed before preflight: {path}")
        size = opened_info.st_size
        if size < 8:
            raise Hy3MTPLoadError(f"{path.name} is too short to be safetensors")

        header_size = struct.unpack(
            "<Q", _pread_exact(fd, 8, 0, label=f"{path.name} header length")
        )[0]
        if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
            raise Hy3MTPLoadError(
                f"{path.name} header size {header_size} is outside the bounded range"
            )
        data_start = 8 + header_size
        if data_start > size:
            raise Hy3MTPLoadError(f"{path.name} has a truncated header")
        header_bytes = _pread_exact(
            fd,
            header_size,
            8,
            label=f"{path.name} header",
        )
        try:
            header = json.loads(
                header_bytes,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            RecursionError,
        ) as exc:
            raise Hy3MTPLoadError(
                f"{path.name} has an invalid safetensors header: {exc}"
            ) from exc
        if not isinstance(header, dict):
            raise Hy3MTPLoadError(f"{path.name} safetensors header must be an object")
        if len(header) > _MAX_SAFETENSORS_TENSORS + 1:
            raise Hy3MTPLoadError(f"{path.name} has too many tensor entries")

        metadata = header.get("__metadata__")
        if not isinstance(metadata, dict):
            raise Hy3MTPLoadError(f"{path.name} has a malformed __metadata__ block")
        revision = metadata.get("source_revision")
        if not isinstance(revision, str):
            raise Hy3MTPLoadError(
                f"{path.name} source_revision metadata must be a string"
            )
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise Hy3MTPLoadError(f"{path.name} has a malformed __metadata__ block")
        source_repo = metadata.get("source_repo")
        if source_repo != HY3_MTP_SOURCE_REPO:
            raise Hy3MTPLoadError(
                f"{path.name} source_repo is {source_repo!r}; "
                f"expected {HY3_MTP_SOURCE_REPO!r}"
            )
        if revision != expected_revision:
            raise Hy3MTPLoadError(
                f"{path.name} was packaged from revision {revision!r}; "
                f"expected {expected_revision!r}"
            )

        ranges: list[tuple[int, int, str]] = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name:
                raise Hy3MTPLoadError(f"{path.name} has an invalid tensor name")
            if not isinstance(tensor, dict) or set(tensor) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise Hy3MTPLoadError(
                    f"{path.name} tensor {name!r} has invalid metadata keys"
                )
            offsets = tensor["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
            ):
                raise Hy3MTPLoadError(
                    f"{path.name} tensor {name!r} has invalid data offsets"
                )
            start, end = offsets
            if start < 0 or end < start:
                raise Hy3MTPLoadError(
                    f"{path.name} tensor {name!r} has an invalid data range"
                )
            expected_bytes = _tensor_payload_bytes(
                tensor["dtype"], tensor["shape"], path=path, name=name
            )
            if end - start != expected_bytes:
                raise Hy3MTPLoadError(
                    f"{path.name} tensor {name!r} has a dtype/shape byte-count "
                    f"mismatch: range={end - start}, expected={expected_bytes}"
                )
            ranges.append((start, end, name))

        if not ranges:
            raise Hy3MTPLoadError(f"{path.name} contains no tensors")
        payload_bytes = size - data_start
        cursor = 0
        for start, end, name in sorted(ranges):
            if start < cursor:
                raise Hy3MTPLoadError(
                    f"{path.name} has overlapping tensor data at {name!r}"
                )
            if start != cursor:
                raise Hy3MTPLoadError(
                    f"{path.name} tensor data is not contiguous at {name!r}"
                )
            if end > payload_bytes:
                raise Hy3MTPLoadError(
                    f"{path.name} tensor {name!r} extends beyond the file"
                )
            cursor = end
        if cursor != payload_bytes:
            raise Hy3MTPLoadError(
                f"{path.name} has trailing data after the final tensor"
            )

        final_info = os.fstat(fd)
        if not _unchanged_while_held(opened_info, final_info):
            raise Hy3MTPLoadError(f"Hy3 MTP artifact changed during preflight: {path}")
        return payload_bytes
    except OSError as exc:
        raise Hy3MTPLoadError(f"cannot read Hy3 MTP artifact {path}: {exc}") from exc


def _same_file_contents(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _unchanged_while_held(before: os.stat_result, after: os.stat_result) -> bool:
    if _same_file_contents(before, after):
        return True
    # Replacing the pathname unlinks the still-open old inode on filesystems
    # such as APFS, which changes only ctime/nlink. The descriptor continues to
    # identify the exact bytes that were preflighted and intentionally remains
    # the load source.
    return (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        and before.st_nlink > 0
        and after.st_nlink == 0
    )


def _selected_artifact_filenames(precision: str) -> tuple[str, ...]:
    return (
        (HY3_MTP_BF16_FILE,)
        if precision == "bf16"
        else (HY3_MTP_RESIDENTS_FILE, HY3_MTP_EXPERTS_FILE)
    )


def _validate_artifact_selection(precision: str, expected_revision: str) -> None:
    if precision not in HY3_MTP_PRECISIONS:
        raise Hy3MTPLoadError(
            f"unsupported Hy3 MTP precision {precision!r}; "
            f"choose one of {HY3_MTP_PRECISIONS}"
        )
    if not isinstance(expected_revision, str) or not expected_revision:
        raise Hy3MTPLoadError("expected Hy3 MTP source revision must be non-empty")


@contextlib.contextmanager
def open_verified_hy3_mtp_artifacts(
    artifact_dir: Path | str,
    *,
    precision: str = HY3_MTP_DEFAULT_PRECISION,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
) -> Iterator[VerifiedHy3MTPArtifacts]:
    """Open, structurally verify, and retain the exact selected artifact inodes.

    Verification is bounded to safetensors headers and revision metadata; it is
    not cryptographic payload authentication. Consumers must pass the yielded
    file objects directly to ``mx.load`` and materialize arrays before exit.
    """

    _validate_artifact_selection(precision, expected_revision)
    root = Path(artifact_dir).expanduser().resolve()
    opened: dict[str, tuple[BinaryIO, os.stat_result]] = {}
    payload_bytes = 0
    with contextlib.ExitStack() as stack:
        for filename in _selected_artifact_filenames(precision):
            path = root / filename
            try:
                path_info = os.lstat(path)
            except FileNotFoundError as exc:
                raise Hy3MTPLoadError(f"missing Hy3 MTP artifact {path}") from exc
            except OSError as exc:
                raise Hy3MTPLoadError(
                    f"cannot inspect Hy3 MTP artifact {path}: {exc}"
                ) from exc
            if not stat.S_ISREG(path_info.st_mode):
                raise Hy3MTPLoadError(
                    f"Hy3 MTP artifact must be a regular file: {path}"
                )

            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise Hy3MTPLoadError(
                    f"Hy3 MTP artifact must be a regular file: {path}: {exc}"
                ) from exc
            try:
                opened_info = os.fstat(fd)
                if not stat.S_ISREG(opened_info.st_mode) or (
                    opened_info.st_dev,
                    opened_info.st_ino,
                ) != (path_info.st_dev, path_info.st_ino):
                    raise Hy3MTPLoadError(
                        "Hy3 MTP artifact was replaced or is not a regular file: "
                        f"{path}"
                    )
                file = os.fdopen(fd, "rb")
            except Exception:
                os.close(fd)
                raise
            stack.enter_context(file)
            payload_bytes += _preflight_safetensors_payload_held(
                path, file.fileno(), opened_info, expected_revision
            )
            file.seek(0)
            opened[filename] = (file, opened_info)

        verified = VerifiedHy3MTPArtifacts(
            root=root,
            precision=precision,
            payload_bytes=payload_bytes,
            source_revision=expected_revision,
            files=MappingProxyType(
                {filename: file for filename, (file, _info) in opened.items()}
            ),
        )
        try:
            yield verified
        finally:
            for filename, (file, opened_info) in opened.items():
                if not _unchanged_while_held(opened_info, os.fstat(file.fileno())):
                    raise Hy3MTPLoadError(
                        f"Hy3 MTP artifact changed while in use: {root / filename}"
                    )


def preflight_hy3_mtp_artifacts(
    artifact_dir: Path | str,
    *,
    precision: str = HY3_MTP_DEFAULT_PRECISION,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
) -> int:
    """Validate selected artifacts and return their resident tensor bytes.

    This performs bounded header-only reads and never imports or allocates MLX
    arrays.  Q4 conservatively charges the complete residents and experts file
    payloads, including pass-through tensors that the weight loader ignores.
    """

    with open_verified_hy3_mtp_artifacts(
        artifact_dir,
        precision=precision,
        expected_revision=expected_revision,
    ) as verified:
        return verified.payload_bytes


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise Hy3MTPLoadError(f"{path} has a malformed __metadata__ block")
    return {str(key): str(value) for key, value in metadata.items()}


def _require_revision(path: Path, expected_revision: str) -> None:
    metadata = read_safetensors_metadata(path)
    revision = metadata.get("source_revision")
    if revision != expected_revision:
        raise Hy3MTPLoadError(
            f"{path.name} was packaged from revision {revision!r}; "
            f"expected {expected_revision!r}"
        )


def _layer_prefix(args: Any) -> str:
    return f"model.layers.{int(args.num_hidden_layers)}."


def expected_resident_names(args: Any) -> set[str]:
    prefix = _layer_prefix(args)
    names = {
        prefix + base + "." + leaf
        for base in _QUANTIZED_RESIDENT_BASES
        for leaf in _QUANT_LEAVES
    }
    names.update(prefix + suffix for suffix in _BF16_RESIDENT_SUFFIXES)
    names.update(prefix + suffix for suffix in _F32_RESIDENT_SUFFIXES)
    return names


def expected_expert_names(args: Any) -> set[str]:
    prefix = _layer_prefix(args)
    return {
        f"{prefix}mlp.experts.{expert}.{projection}.{leaf}"
        for expert in range(int(args.num_experts))
        for projection in _EXPERT_PROJECTIONS
        for leaf in _QUANT_LEAVES
    }


def expected_bf16_names(args: Any) -> set[str]:
    """The full layer-80 inventory of the bit-exact BF16 artifact.

    Every tensor is a single ``.weight`` leaf (the projections that the Q4
    artifacts store quantized are plain BF16 here), plus the F32
    ``mlp.expert_bias`` and one BF16 ``.weight`` per routed expert
    projection.
    """

    prefix = _layer_prefix(args)
    names = {prefix + base + ".weight" for base in _QUANTIZED_RESIDENT_BASES}
    names.update(prefix + suffix for suffix in _BF16_RESIDENT_SUFFIXES)
    names.update(prefix + suffix for suffix in _F32_RESIDENT_SUFFIXES)
    names.update(
        f"{prefix}mlp.experts.{expert}.{projection}.weight"
        for expert in range(int(args.num_experts))
        for projection in _EXPERT_PROJECTIONS
    )
    return names


def _resident_target(suffix: str) -> str:
    """Map one layer-80 artifact suffix to its Hy3MTPLayer parameter path."""

    if suffix == "mlp.expert_bias":
        # The pinned artifact stores the router correction bias under the
        # router module; the source checkpoint keeps it on the MLP.
        return "mtp_block.mlp.router.expert_bias"
    if suffix.split(".", 1)[0] in _HEAD_LOCAL_MODULES:
        return suffix
    return "mtp_block." + suffix


def _validate_leaf_dtype(name: str, value: Any, mx: Any) -> None:
    if name.endswith((".scales", ".biases")):
        if value.dtype != mx.bfloat16:
            raise Hy3MTPLoadError(f"{name} must be bfloat16, found {value.dtype}")
    elif name.endswith("mlp.expert_bias"):
        if value.dtype != mx.float32:
            raise Hy3MTPLoadError(f"{name} must be float32, found {value.dtype}")
    elif name.endswith(".weight"):
        base = name.rsplit(".", 1)[0]
        quantized = (
            any(base.endswith(candidate) for candidate in _QUANTIZED_RESIDENT_BASES)
            or ".mlp.experts." in name
        )
        wanted = mx.uint32 if quantized else mx.bfloat16
        if value.dtype != wanted:
            raise Hy3MTPLoadError(f"{name} must be {wanted}, found {value.dtype}")


def _require_verified_artifacts(
    verified: VerifiedHy3MTPArtifacts,
    artifact_dir: Path,
    *,
    precision: str,
    expected_revision: str,
) -> Mapping[str, BinaryIO]:
    if not isinstance(verified, VerifiedHy3MTPArtifacts):
        raise Hy3MTPLoadError("invalid borrowed Hy3 MTP artifact handle")
    if verified.root != artifact_dir:
        raise Hy3MTPLoadError(
            "borrowed Hy3 MTP artifact root does not match artifact_dir"
        )
    if verified.precision != precision:
        raise Hy3MTPLoadError(
            f"borrowed Hy3 MTP precision is {verified.precision!r}; "
            f"expected {precision!r}"
        )
    if verified.source_revision != expected_revision:
        raise Hy3MTPLoadError(
            f"borrowed Hy3 MTP revision is {verified.source_revision!r}; "
            f"expected {expected_revision!r}"
        )
    expected_files = set(_selected_artifact_filenames(precision))
    if set(verified.files) != expected_files:
        raise Hy3MTPLoadError("borrowed Hy3 MTP artifact file set is invalid")
    if any(file.closed for file in verified.files.values()):
        raise Hy3MTPLoadError("borrowed Hy3 MTP artifact handle is closed")
    return verified.files


def load_hy3_mtp_weights(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    mx_module: Any | None = None,
    verified_artifacts: VerifiedHy3MTPArtifacts | None = None,
) -> dict[str, Any]:
    """Read and validate both layer-80 artifacts into module-path weights.

    Residents come only from the residents artifact and routed experts only
    from the expert artifact (whose BF16 resident pass-through copies are
    ignored).  Expert projections are stacked into ``switch_mlp`` tensors of
    shape ``[num_experts, ...]``.
    """

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if verified_artifacts is None:
        with open_verified_hy3_mtp_artifacts(
            artifact_dir,
            precision="q4",
            expected_revision=expected_revision,
        ) as verified:
            return load_hy3_mtp_weights(
                artifact_dir,
                args,
                expected_revision=expected_revision,
                mx_module=mx,
                verified_artifacts=verified,
            )
    files = _require_verified_artifacts(
        verified_artifacts,
        artifact_dir,
        precision="q4",
        expected_revision=expected_revision,
    )
    residents_path = artifact_dir / HY3_MTP_RESIDENTS_FILE
    experts_path = artifact_dir / HY3_MTP_EXPERTS_FILE

    prefix = _layer_prefix(args)
    mapped: dict[str, Any] = {}

    residents_file = files[HY3_MTP_RESIDENTS_FILE]
    residents_file.seek(0)
    residents = mx.load(residents_file, format="safetensors")
    expected_residents = expected_resident_names(args)
    missing = expected_residents - set(residents)
    extra = set(residents) - expected_residents
    if missing:
        raise Hy3MTPLoadError(
            f"{residents_path.name} is missing tensors: {sorted(missing)[:4]}"
        )
    if extra:
        raise Hy3MTPLoadError(
            f"{residents_path.name} has unexpected tensors: {sorted(extra)[:4]}"
        )
    for name, value in residents.items():
        _validate_leaf_dtype(name, value, mx)
        mapped["layers.0." + _resident_target(name[len(prefix) :])] = value

    experts_file = files[HY3_MTP_EXPERTS_FILE]
    experts_file.seek(0)
    experts = mx.load(experts_file, format="safetensors")
    expected_experts = expected_expert_names(args)
    missing = expected_experts - set(experts)
    if missing:
        raise Hy3MTPLoadError(
            f"{experts_path.name} is missing expert tensors: {sorted(missing)[:4]}"
        )
    unexpected = {
        name for name in set(experts) - expected_experts if ".mlp.experts." in name
    }
    if unexpected:
        raise Hy3MTPLoadError(
            f"{experts_path.name} has unexpected expert tensors: "
            f"{sorted(unexpected)[:4]}"
        )
    num_experts = int(args.num_experts)
    reference_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
    for projection in _EXPERT_PROJECTIONS:
        for leaf in _QUANT_LEAVES:
            values = []
            for expert in range(num_experts):
                name = f"{prefix}mlp.experts.{expert}.{projection}.{leaf}"
                value = experts[name]
                _validate_leaf_dtype(name, value, mx)
                shape = tuple(int(dim) for dim in value.shape)
                reference = reference_shapes.setdefault((projection, leaf), shape)
                if shape != reference:
                    raise Hy3MTPLoadError(
                        f"{name} shape {shape} differs from expert 0 {reference}"
                    )
                values.append(value)
            mapped[f"layers.0.mtp_block.mlp.switch_mlp.{projection}.{leaf}"] = mx.stack(
                values
            )
    mx.eval(mapped)
    return mapped


def load_hy3_mtp_bf16_weights(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    mx_module: Any | None = None,
    verified_artifacts: VerifiedHy3MTPArtifacts | None = None,
) -> dict[str, Any]:
    """Read and validate the bit-exact BF16 layer-80 artifact.

    Fail-closed like the Q4 loader: the file must carry exactly the expected
    tensor inventory at the expected source revision, every tensor BF16
    except the F32 router correction bias.  Expert projections are stacked
    into non-quantized ``switch_mlp`` tensors of shape ``[num_experts, ...]``.
    """

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if verified_artifacts is None:
        with open_verified_hy3_mtp_artifacts(
            artifact_dir,
            precision="bf16",
            expected_revision=expected_revision,
        ) as verified:
            return load_hy3_mtp_bf16_weights(
                artifact_dir,
                args,
                expected_revision=expected_revision,
                mx_module=mx,
                verified_artifacts=verified,
            )
    files = _require_verified_artifacts(
        verified_artifacts,
        artifact_dir,
        precision="bf16",
        expected_revision=expected_revision,
    )
    path = artifact_dir / HY3_MTP_BF16_FILE

    prefix = _layer_prefix(args)
    artifact_file = files[HY3_MTP_BF16_FILE]
    artifact_file.seek(0)
    tensors = mx.load(artifact_file, format="safetensors")
    expected = expected_bf16_names(args)
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    if missing:
        raise Hy3MTPLoadError(f"{path.name} is missing tensors: {sorted(missing)[:4]}")
    if extra:
        raise Hy3MTPLoadError(
            f"{path.name} has unexpected tensors: {sorted(extra)[:4]}"
        )

    mapped: dict[str, Any] = {}
    for name, value in tensors.items():
        if name.endswith("mlp.expert_bias"):
            if value.dtype != mx.float32:
                raise Hy3MTPLoadError(f"{name} must be float32, found {value.dtype}")
        elif value.dtype != mx.bfloat16:
            raise Hy3MTPLoadError(f"{name} must be bfloat16, found {value.dtype}")
        if ".mlp.experts." not in name:
            mapped["layers.0." + _resident_target(name[len(prefix) :])] = value

    num_experts = int(args.num_experts)
    for projection in _EXPERT_PROJECTIONS:
        values = []
        reference: tuple[int, ...] | None = None
        for expert in range(num_experts):
            name = f"{prefix}mlp.experts.{expert}.{projection}.weight"
            value = tensors[name]
            shape = tuple(int(dim) for dim in value.shape)
            if reference is None:
                reference = shape
            elif shape != reference:
                raise Hy3MTPLoadError(
                    f"{name} shape {shape} differs from expert 0 {reference}"
                )
            values.append(value)
        mapped[f"layers.0.mtp_block.mlp.switch_mlp.{projection}.weight"] = mx.stack(
            values
        )
    mx.eval(mapped)
    return mapped


def _quantization_spec_for(
    path: str,
    module: Any,
    weights: dict[str, Any],
    *,
    group_size: int,
) -> bool | dict[str, Any]:
    if not hasattr(module, "to_quantized"):
        return False
    packed = weights.get(f"{path}.weight")
    scales = weights.get(f"{path}.scales")
    if packed is None or scales is None:
        return False
    logical_in = int(module.weight.shape[-1])
    packed_words = int(packed.shape[-1])
    bits, remainder = divmod(packed_words * 32, logical_in)
    if remainder or bits not in {4, 8}:
        raise Hy3MTPLoadError(
            f"{path}.weight packs {packed_words} words for {logical_in} inputs; "
            "cannot derive a 4/8-bit affine layout"
        )
    derived_group, remainder = divmod(logical_in, int(scales.shape[-1]))
    if remainder or derived_group != group_size:
        raise Hy3MTPLoadError(
            f"{path}.scales implies group size {derived_group}; expected {group_size}"
        )
    return {"bits": bits, "group_size": group_size, "mode": "affine"}


def build_hy3_mtp_module(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    group_size: int = 64,
    precision: str = HY3_MTP_DEFAULT_PRECISION,
    shared_kernel: str = "stock",
    shared_kernel_depth: int = 3,
    verified_artifacts: VerifiedHy3MTPArtifacts | None = None,
) -> Any:
    """Construct, strictly load, and evaluate the Hy3 NextN head.

    ``precision="bf16"`` (default) builds the entire head plain BF16 from the
    bit-exact ``layer80-bf16.safetensors`` artifact — no quantized modules
    anywhere (~7.5 GB resident).  ``precision="q4"`` keeps the pinned
    quantized artifacts and behavior (~1.94 GiB expert bank).
    """

    import mlx.core as mx
    import mlx.nn as nn

    from .models.hy3_mlx import Hy3MTP

    if precision not in HY3_MTP_PRECISIONS:
        raise Hy3MTPLoadError(
            f"unsupported Hy3 MTP precision {precision!r}; "
            f"choose one of {HY3_MTP_PRECISIONS}"
        )
    if shared_kernel not in {"stock", "metal-exact"}:
        raise Hy3MTPLoadError(
            "unsupported Hy3 MTP shared kernel "
            f"{shared_kernel!r}; choose 'stock' or 'metal-exact'"
        )
    if (
        isinstance(shared_kernel_depth, bool)
        or not isinstance(shared_kernel_depth, int)
        or shared_kernel_depth < 1
    ):
        raise Hy3MTPLoadError("shared_kernel_depth must be a positive integer")
    if shared_kernel != "stock" and precision != "bf16":
        raise Hy3MTPLoadError("metal-exact shared kernel requires the BF16 MTP head")
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if verified_artifacts is None:
        with open_verified_hy3_mtp_artifacts(
            artifact_dir,
            precision=precision,
            expected_revision=expected_revision,
        ) as verified:
            return build_hy3_mtp_module(
                artifact_dir,
                args,
                expected_revision=expected_revision,
                group_size=group_size,
                precision=precision,
                shared_kernel=shared_kernel,
                shared_kernel_depth=shared_kernel_depth,
                verified_artifacts=verified,
            )
    _require_verified_artifacts(
        verified_artifacts,
        artifact_dir,
        precision=precision,
        expected_revision=expected_revision,
    )
    if precision == "bf16":
        weights = load_hy3_mtp_bf16_weights(
            artifact_dir,
            args,
            expected_revision=expected_revision,
            verified_artifacts=verified_artifacts,
        )
        mtp = Hy3MTP(args, num_mtp_layers=1)
    else:
        weights = load_hy3_mtp_weights(
            artifact_dir,
            args,
            expected_revision=expected_revision,
            verified_artifacts=verified_artifacts,
        )
        mtp = Hy3MTP(args, num_mtp_layers=1)

        def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
            return _quantization_spec_for(path, module, weights, group_size=group_size)

        nn.quantize(
            mtp,
            group_size=group_size,
            bits=4,
            mode="affine",
            class_predicate=class_predicate,
        )
    mtp.eval()
    try:
        mtp.load_weights(list(weights.items()), strict=True)
    except Exception as exc:
        raise Hy3MTPLoadError(f"Hy3 MTP weight validation failed: {exc}") from exc
    if shared_kernel == "metal-exact":
        from .hy3_mtp_shared_gate_up import install_depth_gated_mtp_shared_mlp

        try:
            install_depth_gated_mtp_shared_mlp(
                mtp,
                target_depth=shared_kernel_depth,
            )
        except Exception as exc:
            raise Hy3MTPLoadError(
                f"Hy3 MTP shared-kernel installation failed: {exc}"
            ) from exc
    mx.eval(mtp.parameters())
    logger.info(
        "[Hy3 MTP] loaded %d tensors (%s, shared=%s at fixed depth %d) from %s",
        len(weights),
        precision,
        shared_kernel,
        shared_kernel_depth,
        Path(artifact_dir).expanduser(),
    )
    return mtp


def inject_hy3_streamed_mtp_support(
    model: Any,
    artifact_dir: Path | str | None,
    config: dict[str, Any],
    contract: Any | None = None,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    mtp_precision: str = HY3_MTP_DEFAULT_PRECISION,
    shared_kernel: str = "stock",
    shared_kernel_depth: int = 3,
    mtp_module: Any | None = None,
) -> bool:
    """Attach layer-80 NextN speculative support to a streamed Hy3 model.

    The patched model exposes the same ``__call__`` / ``mtp_forward`` /
    ``mtp_update_cache`` / ``make_mtp_cache`` surface as the other mtplx MTP
    backends, so the existing exact rejection-sampling generate loops drive
    it unchanged.  ``mtp_verify_width`` tells the streamed runtime how wide a
    decode-side verify batch can be so expert routing keeps training the
    persistent decode hot set.

    ``mtp_precision="bf16"`` (default) loads the bit-exact BF16 head
    (~7.5 GB resident — budget it against the expert cache);
    ``"q4"`` loads the pinned quantized artifacts (~1.94 GiB expert bank).
    """

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    if artifact_dir is None:
        raise Hy3MTPLoadError("streamed Hy3 MTP requires an artifact directory")
    if str(config.get("model_type") or "") != "hy_v3":
        raise Hy3MTPLoadError("streamed MTP injection supports hy_v3 only")
    args = getattr(model, "args", None)
    if args is None:
        raise Hy3MTPLoadError("streamed Hy3 model exposes no ModelArgs")
    declared = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
    if declared < 1:
        raise Hy3MTPLoadError(
            "model config declares no NextN predictor layer; refusing to "
            "attach a layer-80 head it never trained"
        )

    mtp = mtp_module
    if mtp is None:
        mtp = build_hy3_mtp_module(
            artifact_dir,
            args,
            expected_revision=expected_revision,
            precision=mtp_precision,
            shared_kernel=shared_kernel,
            shared_kernel_depth=shared_kernel_depth,
        )
    original_outer_class = model.__class__

    class _MTPLXStreamedHy3Model(original_outer_class):
        # Primary token plus one drafted token per NextN layer.  Consumed by
        # MTPLXRuntime to classify short verify batches as decode routing.
        mtp_verify_width = 1 + len(mtp.layers)

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            hidden_variant: str | None = None,
            emit_logits: bool = True,
            logits_keep: int | None = None,
        ):
            if hidden_variant not in {None, "pre_norm", "post_norm"}:
                raise ValueError(
                    "streamed Hy3 MTP supports pre_norm or post_norm hidden variants"
                )
            post_norm, pre_norm = self.model(inputs, cache, return_pre_norm=True)
            logits = None
            if emit_logits:
                head_input = post_norm
                if logits_keep is not None:
                    keep = int(logits_keep)
                    if keep < 1:
                        raise ValueError("logits_keep must be positive when supplied")
                    head_input = head_input[:, -keep:, :]
                # Cast logits, not the head input: fp32 x BF16 matmul
                # materializes an fp32 weight copy per call (see
                # Hy3ForCausalLM.__call__). _logits_head() (inherited from the
                # base Model) applies T2a's optional trunk-head quant here so it
                # reaches the MTP verify, not only the AR path.
                logits = self._logits_head()(head_input)
                if self.args.enable_lm_head_fp32:
                    logits = logits.astype(mx.float32)
            if not return_hidden:
                return logits
            # Gate v1/v2 measured the head prefers the POST-norm trunk
            # hidden (acceptance 0.208 post vs 0.148 pre), matching
            # deepseek_mtp_patch.py and the mtp_patch.py contract default.
            # pre_norm stays selectable for A/B probing.
            hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
            return logits, hidden

        def _draft_lm_head(self):
            """T1c: a quantized COPY of the trunk head for the DRAFT projection only.
            OUTPUT-LOSSLESS — the target verify uses self.lm_head at full precision;
            the draft head only shapes proposals, so quantizing it can move only the
            acceptance rate, never the emitted distribution. Off by default; enable
            with MTPLX_HY3_DRAFT_LM_HEAD_BITS=4 (or 8). Built once, then cached."""
            bits = int(os.environ.get("MTPLX_HY3_DRAFT_LM_HEAD_BITS", "0") or "0")
            if bits <= 0:
                return self.lm_head
            cached = getattr(self, "_mtplx_draft_lm_head", None)
            if cached is None:
                import mlx.core as mx
                import mlx.nn as nn

                cached = nn.QuantizedLinear.from_linear(
                    self.lm_head, group_size=64, bits=bits
                )
                mx.eval(cached.parameters())
                self._mtplx_draft_lm_head = cached
            return cached

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str | None = "post_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError(
                    "streamed Hy3 MTP supports embedding_hidden concat order only"
                )
            depth = 0 if mtp_depth is None else max(int(mtp_depth) - 1, 0)
            depth %= len(self.mtp.layers)
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = (
                    mtp_cache[depth] if isinstance(mtp_cache, list) else mtp_cache
                )
            logits, hidden = self.mtp.layers[depth](
                next_token_ids,
                hidden_states,
                embed_tokens=self.model.embed_tokens,
                lm_head=self._draft_lm_head(),
                cache=layer_cache,
            )
            if not return_hidden:
                return logits
            return logits, hidden

        def configure_mtp_execution_depth(
            self, depth: int | None
        ) -> tuple[str, ...]:
            """Swap depth-gated shared operators once before generation."""

            modes = []
            for layer in self.mtp.layers:
                shared = layer.mtp_block.mlp.shared_mlp
                configure = getattr(shared, "configure_depth", None)
                if callable(configure):
                    modes.append(str(configure(depth)))
            return tuple(modes)

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                position_offset=position_offset,
                mtp_depth=mtp_depth,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache() for _layer in self.mtp.layers]

    model.mtp = mtp
    model.__class__ = _MTPLXStreamedHy3Model
    logger.info(
        "[Hy3 MTP inject] streamed NextN head attached (verify width %d)",
        _MTPLXStreamedHy3Model.mtp_verify_width,
    )
    return True
