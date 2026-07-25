#!/usr/bin/env python
"""Build a mixed-precision "official recipe" expert bank (issue #51, M1).

Copies Tencent's AngelSlim per-layer allocation (down = IQ3_XXS everywhere;
gate/up = IQ1_M or IQ2_XXS per layer, from
``research/angelslim_mimic/official-recipe-map.json``) using our *servable*
encodings -- t158 for IQ1_M, 2-bit affine for IQ2_XXS, 3-bit affine for
IQ3_XXS.  Reads the bf16 source tensors with pread range-reads (never a whole
7 GB shard), encodes one expert at a time, and writes a single page-aligned
sidecar bin plus a mixed-official manifest emitted atomically only after the
whole selection is on disk and structurally verified.

Resumable: a JSON journal (atomic .tmp + os.replace) records the byte cursor
and the number of durable records; a crashed run truncates its torn tail,
rehashes the completed records straight from the bin and restarts at the
first missing one, producing byte-identical output.

``--limit-layers`` / ``--limit-experts`` bound the selection for smoke runs
(and make the converter sharded-friendly -- an orchestrator can drive
disjoint layer ranges).  ``--verify-sample N`` re-reads N written records and
requires a bitwise round trip against a fresh encode of the bf16 source.

CPU-only.  BUILD + smoke lane; the full ~87 GiB burn is governor-gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from mtplx.expert_mixed_official import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    DEFAULT_SIDECAR_NAME,
    MIXED_ALIGNMENT,
    MIXED_GROUP_SIZE,
    MIXED_MANIFEST_FORMAT,
    PROJECTIONS,
    TIERS,
    align_up,
    build_layer_tier_map,
    canonical_json,
    decode_projection,
    encode_projection,
    manifest_digest,
    plan_record_segments,
    quantization_block,
    source_recipe_block,
    tier_for_projection,
    validate_mixed_manifest,
)
from research.iq_transcode.st_range import _load_header, read_tensor_fp32  # noqa: E402

_JOURNAL_FORMAT = "mtplx-mixofficial-journal-v1"


class ConvertError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _parse_ints(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(part) for part in text.split(",") if part.strip()}


def _sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _load_coverage(manifest_path: Path) -> dict[str, Any]:
    """Read the authoritative expert manifest for record coverage/provenance."""

    with manifest_path.open("rb") as handle:
        raw = json.load(handle)
    if raw.get("format") != MIXED_MANIFEST_FORMAT:
        raise ConvertError(
            f"coverage manifest {manifest_path} is not {MIXED_MANIFEST_FORMAT}"
        )
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ConvertError("coverage manifest has no records")
    keys = sorted((int(r["layer"]), int(r["expert"])) for r in records)
    if len(keys) != len(set(keys)):
        raise ConvertError("coverage manifest has duplicate (layer, expert) records")
    return {
        "keys": keys,
        "moe_layers": sorted({layer for layer, _ in keys}),
        "source_repo": str(raw.get("source_repo", "unknown")),
        "source_revision": str(raw.get("source_revision", "unknown")),
        "manifest_sha256": raw.get("manifest_sha256"),
    }


class _Bf16Source:
    """Resolve and range-read per-expert bf16 projection tensors."""

    def __init__(self, root: Path) -> None:
        self.root = root
        index_path = root / "model.safetensors.index.json"
        with index_path.open("rb") as handle:
            self.weight_map = json.load(handle)["weight_map"]

    def tensor_name(self, layer: int, expert: int, projection: str) -> str:
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def _shard_for(self, name: str) -> Path:
        shard = self.weight_map.get(name)
        if shard is None:
            raise ConvertError(f"bf16 source has no tensor {name}")
        return self.root / shard

    def read(self, layer: int, expert: int, projection: str) -> np.ndarray:
        name = self.tensor_name(layer, expert, projection)
        arr, _meta = read_tensor_fp32(str(self._shard_for(name)), name)
        return arr

    def dims(self, layer: int, expert: int, projection: str) -> tuple[int, int]:
        """Tensor shape from the shard header alone (no data bytes read)."""

        name = self.tensor_name(layer, expert, projection)
        _start, header = _load_header(str(self._shard_for(name)))
        shape = tuple(int(v) for v in header[name]["shape"])
        if len(shape) != 2:
            raise ConvertError(f"tensor {name} is not 2-D: {shape}")
        return shape


def _selection_fingerprint(
    keys: Sequence[tuple[int, int]],
    layer_tier_map: dict[int, dict[str, str]],
    coverage_sha: str | None,
    map_sha256: str,
) -> str:
    payload = canonical_json(
        {
            "keys": [list(k) for k in keys],
            "layer_tier_map": {
                str(layer): tiers for layer, tiers in sorted(layer_tier_map.items())
            },
            "coverage_sha256": coverage_sha,
            "map_sha256": map_sha256,
            "alignment": MIXED_ALIGNMENT,
        }
    )
    return hashlib.sha256(payload).hexdigest()


def _encode_record(
    source: _Bf16Source,
    layer: int,
    expert: int,
    layer_tiers: dict[str, str],
) -> tuple[bytes, list[dict[str, Any]], dict[str, tuple[int, int]]]:
    """Encode one expert into (record blob, in-record segments, dims)."""

    dims: dict[str, tuple[int, int]] = {}
    leaf_bytes: dict[str, bytes] = {}
    for projection in PROJECTIONS:
        weights = source.read(layer, expert, projection)
        rows, cols = int(weights.shape[0]), int(weights.shape[1])
        dims[projection] = (rows, cols)
        tier = tier_for_projection(layer_tiers, projection)
        for payload in encode_projection(tier, weights, group_size=MIXED_GROUP_SIZE):
            leaf_bytes[f"{projection}.{payload.leaf}"] = payload.data
    planned = plan_record_segments(layer_tiers, dims)
    chunks: list[bytes] = []
    segments: list[dict[str, Any]] = []
    for seg in planned:
        data = leaf_bytes[seg.component]
        if len(data) != seg.length:
            raise ConvertError(
                f"record ({layer}, {expert}) segment {seg.component} encoded "
                f"{len(data)} bytes; layout expects {seg.length}"
            )
        projection, leaf = seg.component.split(".", 1)
        segments.append(
            {
                "component": seg.component,
                "quant_tier": seg.quant_tier,
                "tensor": f"model.layers.{layer}.mlp.switch_mlp.{projection}.{leaf}",
                "shard": DEFAULT_SIDECAR_NAME,
                "offset": seg.offset,  # rebased to absolute by the caller
                "length": seg.length,
                "dtype": seg.dtype,
                "shape": list(seg.shape),
            }
        )
        chunks.append(data)
    return b"".join(chunks), segments, dims


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < length:
        block = os.pread(fd, length - got, offset + got)
        if not block:
            raise ConvertError("short pread on the sidecar bin")
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


# --------------------------------------------------------------------------
# core conversion
# --------------------------------------------------------------------------
def convert_mixed_official(
    *,
    source_manifest: Path,
    recipe_map: Path,
    bf16_root: Path,
    output_root: Path,
    source_repo: str | None = None,
    limit_layers: set[int] | None = None,
    limit_experts: set[int] | None = None,
    resume: bool = False,
    overwrite: bool = False,
    verify_sample: int = 2,
    progress: bool = False,
    fault_after: int | None = None,
    allow_unaligned_segments: bool = False,
) -> dict[str, Any]:
    """Convert (a slice of) the routed bank; return the emitted manifest dict.

    ``fault_after`` is a test-only interruption hook: raise after that many
    records are durably journaled to exercise the resume path.
    ``allow_unaligned_segments`` waives the every-segment-16384-aligned law
    for tiny test fixtures only; the real hy3 dims satisfy it and the CLI
    never waives it.
    """

    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bin_path = output_root / DEFAULT_SIDECAR_NAME
    partial_path = bin_path.with_name(bin_path.name + ".partial")
    manifest_path = output_root / "expert-manifest.json"
    journal_path = output_root / "expert-manifest.json.journal"

    if manifest_path.exists() and not overwrite:
        raise ConvertError(f"mixed-official manifest already exists: {manifest_path}")

    coverage = _load_coverage(Path(source_manifest))
    with Path(recipe_map).open("rb") as handle:
        recipe_bytes = handle.read()
    recipe = json.loads(recipe_bytes)
    map_sha256 = hashlib.sha256(recipe_bytes).hexdigest()

    # The full MoE-layer set is the authority: refuse if the recipe leaves any
    # routed layer unassigned (independent of any smoke --limit-layers).
    layer_tier_map = build_layer_tier_map(recipe, coverage["moe_layers"])

    # blk <-> manifest layer correspondence (derived, not assumed).
    per_layer = recipe.get("per_layer_routed", {})
    recipe_layers = sorted(int(k) for k in per_layer)
    if recipe_layers != coverage["moe_layers"]:
        raise ConvertError(
            "recipe routed layers do not match the manifest MoE layers: "
            f"recipe {recipe_layers[:3]}..{recipe_layers[-3:]} vs manifest "
            f"{coverage['moe_layers'][:3]}..{coverage['moe_layers'][-3:]}"
        )
    blk_layer_offset = 0  # identity: manifest layer L == gguf blk.L

    keys = [
        (layer, expert)
        for layer, expert in coverage["keys"]
        if (limit_layers is None or layer in limit_layers)
        and (limit_experts is None or expert in limit_experts)
    ]
    if not keys:
        raise ConvertError("selection is empty after applying --limit filters")

    source = _Bf16Source(Path(bf16_root))
    fingerprint = _selection_fingerprint(
        keys, layer_tier_map, coverage["manifest_sha256"], map_sha256
    )

    records: list[dict[str, Any]] = []
    cursor = 0
    skip = 0

    # ---- resume from journal ------------------------------------------------
    if resume and journal_path.exists() and partial_path.exists():
        journal = json.loads(journal_path.read_text())
        if (
            journal.get("format") != _JOURNAL_FORMAT
            or journal.get("fingerprint") != fingerprint
        ):
            raise ConvertError("journal does not match this selection; refusing resume")
        skip = int(journal["records_done"])
        cursor = int(journal["cursor"])
        if skip > len(keys):
            raise ConvertError("journal claims more records than the selection")
        # Recompute the layout of completed records and rehash them from the
        # bin so the manifest is rebuilt byte-for-byte. Routed experts share
        # one shape, so the template dims come from the shard header of the
        # first selected record; the final cursor equality check would catch
        # any divergence in aggregate.
        template_dims = {
            projection: source.dims(keys[0][0], keys[0][1], projection)
            for projection in PROJECTIONS
        }
        fd = os.open(partial_path, os.O_RDONLY)
        try:
            recomputed_cursor = 0
            for index in range(skip):
                layer, expert = keys[index]
                offset = align_up(recomputed_cursor, MIXED_ALIGNMENT)
                seg_template = _template_segments(
                    layer, layer_tier_map[layer], template_dims
                )
                logical = sum(seg["length"] for seg in seg_template)
                blob = _pread_exact(fd, offset, logical)
                records.append(
                    _finalize_record(
                        layer, expert, offset, logical, seg_template,
                        hashlib.sha256(blob).hexdigest(),
                    )
                )
                recomputed_cursor = offset + logical
            if recomputed_cursor != cursor:
                raise ConvertError(
                    f"journal cursor {cursor} != recomputed {recomputed_cursor}"
                )
        finally:
            os.close(fd)
        if progress:
            print(f"mixofficial: resuming after {skip}/{len(keys)} records", flush=True)
    elif resume and (journal_path.exists() or partial_path.exists()):
        raise ConvertError("resume needs both the journal and the partial bin")
    elif not resume and (partial_path.exists() or journal_path.exists()):
        if not overwrite:
            raise ConvertError("stale partial/journal present; pass --resume or --overwrite")
        partial_path.unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)

    # ---- write records ------------------------------------------------------
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(partial_path, flags, 0o644)
    started = time.perf_counter()
    try:
        os.ftruncate(fd, cursor)
        for index in range(skip, len(keys)):
            layer, expert = keys[index]
            offset = align_up(cursor, MIXED_ALIGNMENT)
            if offset > cursor:
                os.ftruncate(fd, offset)  # zero-pad the alignment gap
            blob, segments, _dims = _encode_record(
                source, layer, expert, layer_tier_map[layer]
            )
            for seg in segments:
                seg["offset"] += offset  # rebase in-record -> absolute
            _pwrite_all(fd, blob, offset)
            os.fsync(fd)
            records.append(
                _finalize_record(
                    layer, expert, offset, len(blob), segments,
                    hashlib.sha256(blob).hexdigest(),
                )
            )
            cursor = offset + len(blob)
            _save_journal(journal_path, fingerprint, index + 1, cursor)
            if fault_after is not None and (index + 1) >= fault_after:
                raise _Interrupt(f"fault_after={fault_after} reached")
            if progress and (index + 1) % 64 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"mixofficial: {index + 1}/{len(keys)} records "
                    f"({cursor / 1024**2:.0f} MiB, {elapsed:.0f}s)",
                    flush=True,
                )
        os.ftruncate(fd, cursor)
        os.fsync(fd)
    finally:
        os.close(fd)

    # ---- verify EVERYTHING before anything is published ---------------------
    sidecar_sha256 = _sha256_file(partial_path)
    manifest = _build_manifest(
        model_key=DEFAULT_MODEL_KEY,
        source_repo=source_repo or coverage["source_repo"],
        source_revision=coverage["source_revision"],
        layer_tier_map=layer_tier_map,
        recipe=recipe,
        map_sha256=map_sha256,
        blk_layer_offset=blk_layer_offset,
        bf16_root=Path(bf16_root),
        records=records,
        sidecar_size=cursor,
        sidecar_sha256=sidecar_sha256,
    )

    # HARD LAYOUT LAW: every segment offset 16384-aligned. The real hy3 dims
    # guarantee it (every leaf length is a 16384 multiple); tiny fixtures may
    # waive it explicitly, but never through the CLI.
    if not _segments_all_aligned(records, MIXED_ALIGNMENT):
        if not allow_unaligned_segments:
            raise ConvertError(
                "segment offsets violate the 16384-alignment law; this cannot "
                "happen with the real hy3 dims -- refusing to publish"
            )
        require_alignment = None
    else:
        require_alignment = MIXED_ALIGNMENT
    validate_mixed_manifest(
        manifest,
        alignment=MIXED_ALIGNMENT,
        require_segment_alignment=require_alignment,
    )
    if verify_sample:
        _verify_sample(
            partial_path, manifest, source, layer_tier_map, verify_sample, progress
        )

    # ---- publish bin + manifest atomically ----------------------------------
    os.replace(partial_path, bin_path)
    _write_json_atomic(manifest_path, manifest)
    journal_path.unlink(missing_ok=True)
    dir_fd = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    return manifest


class _Interrupt(RuntimeError):
    """Test-only simulated interruption (bin+journal are already durable)."""


def _template_segments(
    layer: int,
    layer_tiers: dict[str, str],
    dims: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Segment template (in-record offsets) for resume; no bf16 data read."""

    planned = plan_record_segments(layer_tiers, dims)
    segments: list[dict[str, Any]] = []
    for seg in planned:
        projection, leaf = seg.component.split(".", 1)
        segments.append(
            {
                "component": seg.component,
                "quant_tier": seg.quant_tier,
                "tensor": f"model.layers.{layer}.mlp.switch_mlp.{projection}.{leaf}",
                "shard": DEFAULT_SIDECAR_NAME,
                "offset": seg.offset,
                "length": seg.length,
                "dtype": seg.dtype,
                "shape": list(seg.shape),
            }
        )
    return segments


def _finalize_record(
    layer: int,
    expert: int,
    offset: int,
    logical: int,
    segments: list[dict[str, Any]],
    sha256: str,
) -> dict[str, Any]:
    # Ensure segment offsets are absolute (resume templates are in-record).
    if segments and segments[0]["offset"] != offset:
        cursor = offset
        for seg in segments:
            seg["offset"] = cursor
            cursor += seg["length"]
    return {
        "layer": layer,
        "expert": expert,
        "logical_bytes": logical,
        "sidecar_offset": offset,
        "sidecar_length": logical,
        "sha256": sha256,
        "segments": segments,
    }


def _pwrite_all(fd: int, blob: bytes, offset: int) -> None:
    view = memoryview(blob)
    position = offset
    while view:
        written = os.pwrite(fd, view, position)
        if written <= 0:
            raise ConvertError("short positional write to the sidecar bin")
        position += written
        view = view[written:]


def _save_journal(path: Path, fingerprint: str, records_done: int, cursor: int) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format": _JOURNAL_FORMAT,
        "fingerprint": fingerprint,
        "records_done": records_done,
        "cursor": cursor,
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _segments_all_aligned(records: list[dict[str, Any]], alignment: int) -> bool:
    for record in records:
        for seg in record["segments"]:
            if seg["offset"] % alignment:
                return False
    return True


def _build_manifest(
    *,
    model_key: str,
    source_repo: str,
    source_revision: str,
    layer_tier_map: dict[int, dict[str, str]],
    recipe: dict[str, Any],
    map_sha256: str,
    blk_layer_offset: int,
    bf16_root: Path,
    records: list[dict[str, Any]],
    sidecar_size: int,
    sidecar_sha256: str,
) -> dict[str, Any]:
    recipe_block = source_recipe_block(recipe, map_sha256)
    recipe_block["gguf_blk_to_manifest_layer_offset"] = blk_layer_offset
    recipe_block["bf16_source"] = bf16_root.name
    quant = quantization_block(layer_tier_map, recipe_block)
    routed_bytes = sum(record["logical_bytes"] for record in records)
    manifest = {
        "format": MIXED_MANIFEST_FORMAT,
        "model_key": model_key,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "quantization": quant,
        "artifact": {
            "tensor_bytes": routed_bytes,
            "resident_tensor_bytes": 0,
            "routed_expert_bytes": routed_bytes,
            "shard_count": 1,
            "record_count": len(records),
        },
        "shards": [
            {
                "name": DEFAULT_SIDECAR_NAME,
                "size": sidecar_size,
                "kind": "sidecar",
                "sha256": sidecar_sha256,
            }
        ],
        "resident_tensors": [],
        "records": records,
        "sidecar": {
            "file": DEFAULT_SIDECAR_NAME,
            "alignment": MIXED_ALIGNMENT,
            "size": sidecar_size,
            "sha256": sidecar_sha256,
        },
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def _verify_sample(
    bin_path: Path,
    manifest: dict[str, Any],
    source: _Bf16Source,
    layer_tier_map: dict[int, dict[str, str]],
    sample: int,
    progress: bool,
) -> None:
    records = manifest["records"]
    if sample >= len(records):
        chosen = list(range(len(records)))
    else:
        step = max(1, len(records) // sample)
        chosen = list(range(0, len(records), step))[:sample]
    fd = os.open(bin_path, os.O_RDONLY)
    try:
        for index in chosen:
            record = records[index]
            layer, expert = record["layer"], record["expert"]
            blob = _pread_exact(fd, record["sidecar_offset"], record["sidecar_length"])
            if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                raise ConvertError(f"verify: record ({layer}, {expert}) hash mismatch")
            _verify_record_roundtrip(
                blob, record, source, layer, expert, layer_tier_map[layer]
            )
            if progress:
                print(f"verified round trip: layer {layer} expert {expert}", flush=True)
    finally:
        os.close(fd)


def _verify_record_roundtrip(
    blob: bytes,
    record: dict[str, Any],
    source: _Bf16Source,
    layer: int,
    expert: int,
    layer_tiers: dict[str, str],
) -> None:
    """Decode the stored record and match a fresh encode+decode of the source."""

    base = record["sidecar_offset"]
    leaves: dict[str, np.ndarray] = {}
    for seg in record["segments"]:
        start = seg["offset"] - base
        view = memoryview(blob)[start : start + seg["length"]]
        dtype = {"U8": "u1", "U16": "<u2", "U32": "<u4", "BF16": "<u2"}[seg["dtype"]]
        leaves[seg["component"]] = np.frombuffer(view, dtype=dtype).reshape(seg["shape"])
    # Re-encode the bf16 source and compare stored bytes + decoded weights.
    fresh_blob, fresh_segments, _dims = _encode_record(
        source, layer, expert, layer_tiers
    )
    if fresh_blob != blob:
        raise ConvertError(
            f"verify: record ({layer}, {expert}) stored bytes != fresh encode"
        )
    for projection in PROJECTIONS:
        tier = tier_for_projection(layer_tiers, projection)
        proj_leaves = {
            leaf: leaves[f"{projection}.{leaf}"]
            for leaf, _dtype in TIERS[tier].leaves
        }
        stored = decode_projection(tier, proj_leaves, group_size=MIXED_GROUP_SIZE)
        weights = source.read(layer, expert, projection)
        fresh_leaf = {}
        for payload in encode_projection(tier, weights, group_size=MIXED_GROUP_SIZE):
            dtype = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BF16": np.uint16}[
                payload.dtype
            ]
            fresh_leaf[payload.leaf] = np.frombuffer(payload.data, dtype=dtype).reshape(
                payload.shape
            )
        reference = decode_projection(tier, fresh_leaf, group_size=MIXED_GROUP_SIZE)
        if not np.array_equal(
            stored.view(np.uint32), reference.view(np.uint32)
        ):
            raise ConvertError(
                f"verify: record ({layer}, {expert}) projection {projection} "
                f"tier {tier} decode is not bitwise-identical"
            )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=home / ".cache/huggingface/hy3-expert-only-mlx-q2/expert-manifest.json",
        help="Authoritative expert manifest for record coverage/provenance.",
    )
    parser.add_argument(
        "--recipe-map",
        type=Path,
        default=_ROOT / "research/angelslim_mimic/official-recipe-map.json",
    )
    parser.add_argument(
        "--bf16-root",
        type=Path,
        default=home / ".cache/huggingface/hy3-mtp-layer80",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=home / ".cache/huggingface/hy3-expert-only-mlx-mixofficial",
    )
    parser.add_argument("--source-repo", default="tencent/Hy3")
    parser.add_argument("--limit-layers", type=str, default="")
    parser.add_argument("--limit-experts", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-sample", type=int, default=2)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = convert_mixed_official(
        source_manifest=args.source_manifest,
        recipe_map=args.recipe_map,
        bf16_root=args.bf16_root,
        output_root=args.output_root,
        source_repo=args.source_repo,
        limit_layers=_parse_ints(args.limit_layers),
        limit_experts=_parse_ints(args.limit_experts),
        resume=args.resume,
        overwrite=args.overwrite,
        verify_sample=args.verify_sample,
        progress=True,
    )
    elapsed = time.perf_counter() - started
    routed = manifest["artifact"]["routed_expert_bytes"]
    print(
        f"mixofficial: wrote {manifest['artifact']['record_count']} records, "
        f"{routed / 1024**2:.1f} MiB in {elapsed:.1f}s -> {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
