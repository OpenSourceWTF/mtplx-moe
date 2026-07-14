from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import mtplx.glm52_mtp_artifact as artifact


PREFIX = "model.layers.78."
SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_glm52_mtp_layer78.py"


@dataclass
class SyntheticFixture:
    source: Path
    output: Path
    producer: Path
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
    shard_names: tuple[str, ...]

    @property
    def config(self) -> artifact.Glm52MtpArtifactConfig:
        return artifact.Glm52MtpArtifactConfig(
            source_root=self.source,
            output_root=self.output,
            producer_root=self.producer,
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    header_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
    trailing: bytes = b"",
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    if header_transform is not None:
        header = header_transform(header)
    encoded = _canonical_json(header)
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + trailing)


def _read_safetensors(path: Path) -> dict[str, bytes]:
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size])
    data_start = 8 + header_size
    return {
        name: raw[
            data_start + value["data_offsets"][0] : data_start
            + value["data_offsets"][1]
        ]
        for name, value in header.items()
        if name != "__metadata__"
    }


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_clean_producer(root: Path) -> None:
    root.mkdir()
    _git("init", "-q", cwd=root)
    _git("config", "user.name", "Artifact Test", cwd=root)
    _git("config", "user.email", "artifact@example.invalid", cwd=root)
    (root / "producer.py").write_text("# synthetic clean producer\n", encoding="utf-8")
    _git("add", "producer.py", cwd=root)
    _git("commit", "-qm", "synthetic producer", cwd=root)


def _pin_synthetic_source(
    monkeypatch: pytest.MonkeyPatch,
    fixture: SyntheticFixture,
    expectations: dict[str, artifact.TensorExpectation],
    *,
    shard_counts: dict[str, int] | None = None,
) -> None:
    pins = {
        path.name: artifact.SourceFilePin(path.stat().st_size, _sha256(path))
        for path in sorted(fixture.source.iterdir())
        if path.is_file()
    }
    monkeypatch.setattr(artifact, "PINNED_SOURCE_FILES", pins)
    monkeypatch.setattr(
        artifact,
        "EXPECTED_SHARD_COUNTS",
        shard_counts
        or {
            fixture.shard_names[0]: 2,
            fixture.shard_names[1]: 1,
        },
    )
    monkeypatch.setattr(
        artifact,
        "expected_glm52_layer78_inventory",
        lambda _config: dict(expectations),
    )
    monkeypatch.setattr(artifact, "EXPECTED_TENSOR_COUNT", len(expectations))
    monkeypatch.setattr(
        artifact,
        "EXPECTED_PAYLOAD_BYTES",
        sum(item.nbytes for item in expectations.values()),
    )
    monkeypatch.setattr(
        artifact,
        "EXPECTED_BF16_COUNT",
        sum(item.dtype == "BF16" for item in expectations.values()),
    )
    monkeypatch.setattr(
        artifact,
        "EXPECTED_F32_COUNT",
        sum(item.dtype == "F32" for item in expectations.values()),
    )


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticFixture:
    source = tmp_path / "glm52-mtp-layer78-source"
    source.mkdir(mode=0o700)
    output = tmp_path / "glm52-mtp-layer78"
    producer = tmp_path / "producer"
    _init_clean_producer(producer)

    tensors = {
        PREFIX + "alpha.weight": ("BF16", (2,), b"\x01\x02\x03\x04"),
        PREFIX + "beta.weight": ("F32", (1,), b"\x05\x06\x07\x08"),
        PREFIX + "gamma.weight": ("BF16", (1, 2), b"\x09\x0a\x0b\x0c"),
    }
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    _write_safetensors(
        source / shard_names[0],
        {name: tensors[name] for name in tuple(tensors)[:2]},
    )
    _write_safetensors(
        source / shard_names[1], {tuple(tensors)[2]: tensors[tuple(tensors)[2]]}
    )
    _write_json(source / "config.json", {"synthetic": True})
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 12},
            "weight_map": {
                tuple(tensors)[0]: shard_names[0],
                tuple(tensors)[1]: shard_names[0],
                tuple(tensors)[2]: shard_names[1],
            },
        },
    )
    fixture = SyntheticFixture(source, output, producer, tensors, shard_names)
    expectations = {
        name: artifact.TensorExpectation(dtype, shape)
        for name, (dtype, shape, _raw) in tensors.items()
    }
    _pin_synthetic_source(monkeypatch, fixture, expectations)
    return fixture


def _repin(monkeypatch: pytest.MonkeyPatch, fixture: SyntheticFixture) -> None:
    expectations = {
        name: artifact.TensorExpectation(dtype, shape)
        for name, (dtype, shape, _raw) in fixture.tensors.items()
    }
    _pin_synthetic_source(monkeypatch, fixture, expectations)


def _rewrite_manifest(root: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    path = root / artifact.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    path.write_bytes(_canonical_json(manifest) + b"\n")


def test_exact_glm52_layer78_inventory() -> None:
    inventory = artifact.expected_glm52_layer78_inventory(
        {
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
            "model_type": "glm_moe_dsa",
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "num_attention_heads": 64,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "n_shared_experts": 1,
        }
    )

    assert len(inventory) == 791
    assert sum(item.nbytes for item in inventory.values()) == 19_905_841_664
    assert sum(item.dtype == "BF16" for item in inventory.values()) == 790
    assert sum(item.dtype == "F32" for item in inventory.values()) == 1
    assert inventory[PREFIX + "eh_proj.weight"] == artifact.TensorExpectation(
        "BF16", (6144, 12288)
    )
    assert inventory[PREFIX + "mlp.experts.255.down_proj.weight"] == (
        artifact.TensorExpectation("BF16", (6144, 2048))
    )


def test_preflight_extract_and_deep_verify_preserve_exact_bits(
    synthetic: SyntheticFixture,
) -> None:
    plan = artifact.preflight_glm52_mtp_layer78(synthetic.config)
    assert plan.tensor_count == 3
    assert plan.payload_bytes == 12
    assert plan.shard_distribution == {
        synthetic.shard_names[0]: 2,
        synthetic.shard_names[1]: 1,
    }

    published = artifact.extract_glm52_mtp_layer78(synthetic.config)

    assert published == synthetic.output
    assert sorted(path.name for path in published.iterdir()) == [
        artifact.ARTIFACT_FILE,
        artifact.MANIFEST_FILE,
    ]
    assert _read_safetensors(published / artifact.ARTIFACT_FILE) == {
        name: raw for name, (_dtype, _shape, raw) in synthetic.tensors.items()
    }
    verified = artifact.verify_glm52_mtp_layer78(published, deep=True)
    assert verified["schema"] == artifact.MANIFEST_SCHEMA
    assert set(verified) == artifact.MANIFEST_KEYS
    assert verified["inventory"]["tensor_count"] == 3
    assert verified["inventory"]["payload_bytes"] == 12
    assert verified["producer"]["clean"] is True
    assert len(verified["artifact"]["tensors"]) == 3


@pytest.mark.parametrize(
    "filename",
    ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"],
)
def test_pinned_source_digest_mismatch_is_rejected(
    synthetic: SyntheticFixture,
    filename: str,
) -> None:
    path = synthetic.source / filename
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(artifact.ArtifactValidationError, match="pinned (size|SHA-256)"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
def test_malformed_json_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    (synthetic.source / filename).write_text("{not-json", encoding="utf-8")
    _repin(monkeypatch, synthetic)

    with pytest.raises(artifact.ArtifactValidationError, match="JSON"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


def test_duplicate_json_key_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (synthetic.source / "config.json").write_text(
        '{"synthetic":true,"synthetic":true}\n', encoding="utf-8"
    )
    _repin(monkeypatch, synthetic)

    with pytest.raises(artifact.ArtifactValidationError, match="duplicate JSON key"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_inventory_name_mismatch_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    index_path = synthetic.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if case == "missing":
        del index["weight_map"][tuple(synthetic.tensors)[0]]
    else:
        index["weight_map"][PREFIX + "unexpected.weight"] = synthetic.shard_names[0]
    _write_json(index_path, index)
    _repin(monkeypatch, synthetic)

    with pytest.raises(artifact.ArtifactValidationError, match="inventory"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


@pytest.mark.parametrize("case", ["dtype", "shape"])
def test_wrong_dtype_or_shape_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    target = tuple(synthetic.tensors)[0]

    def mutate(header: dict[str, object]) -> dict[str, object]:
        tensor = header[target]
        assert isinstance(tensor, dict)
        tensor["dtype" if case == "dtype" else "shape"] = (
            "F32" if case == "dtype" else [1, 2]
        )
        return header

    _write_safetensors(
        synthetic.source / synthetic.shard_names[0],
        {name: synthetic.tensors[name] for name in tuple(synthetic.tensors)[:2]},
        header_transform=mutate,
    )
    _repin(monkeypatch, synthetic)

    with pytest.raises(artifact.ArtifactValidationError, match=case):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


@pytest.mark.parametrize("case", ["overlap", "trailing"])
def test_overlapping_ranges_and_trailing_data_are_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    names = tuple(synthetic.tensors)[:2]

    def overlap(header: dict[str, object]) -> dict[str, object]:
        second = header[names[1]]
        assert isinstance(second, dict)
        second["data_offsets"] = [2, 6]
        return header

    _write_safetensors(
        synthetic.source / synthetic.shard_names[0],
        {name: synthetic.tensors[name] for name in names},
        header_transform=overlap if case == "overlap" else None,
        trailing=b"trailing" if case == "trailing" else b"",
    )
    _repin(monkeypatch, synthetic)

    with pytest.raises(
        artifact.ArtifactValidationError, match="overlap|trailing|contiguous"
    ):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


def test_duplicate_tensor_across_source_shards_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = tuple(synthetic.tensors)[0]
    second = tuple(synthetic.tensors)[2]
    _write_safetensors(
        synthetic.source / synthetic.shard_names[1],
        {second: synthetic.tensors[second], duplicate: synthetic.tensors[duplicate]},
    )
    _repin(monkeypatch, synthetic)

    with pytest.raises(artifact.ArtifactValidationError, match="duplicate tensor"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


def test_source_shard_distribution_is_exact(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_synthetic_source(
        monkeypatch,
        synthetic,
        {
            name: artifact.TensorExpectation(dtype, shape)
            for name, (dtype, shape, _raw) in synthetic.tensors.items()
        },
        shard_counts={synthetic.shard_names[0]: 1, synthetic.shard_names[1]: 2},
    )

    with pytest.raises(artifact.ArtifactValidationError, match="shard distribution"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


def test_short_pread_aborts_without_publication(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = artifact.preflight_glm52_mtp_layer78(synthetic.config)
    monkeypatch.setattr(artifact, "preflight_glm52_mtp_layer78", lambda _config: plan)
    real_pread = artifact.os.pread

    def short_pread(fd: int, count: int, offset: int) -> bytes:
        if count == 4:
            return b""
        return real_pread(fd, count, offset)

    monkeypatch.setattr(artifact.os, "pread", short_pread)

    with pytest.raises(artifact.ArtifactValidationError, match="short read"):
        artifact.extract_glm52_mtp_layer78(synthetic.config)
    assert not synthetic.output.exists()


def test_source_replacement_during_copy_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = artifact._copy_tensor_bytes
    replaced = False

    def replace_after_copy(*args: object, **kwargs: object) -> str:
        nonlocal replaced
        result = original(*args, **kwargs)
        if not replaced:
            path = synthetic.source / synthetic.shard_names[0]
            replacement = path.with_suffix(".replacement")
            shutil.copyfile(path, replacement)
            os.replace(replacement, path)
            replaced = True
        return result

    monkeypatch.setattr(artifact, "_copy_tensor_bytes", replace_after_copy)

    with pytest.raises(artifact.ArtifactValidationError, match="replaced"):
        artifact.extract_glm52_mtp_layer78(synthetic.config)
    assert not synthetic.output.exists()


def test_manifest_schema_and_digest_tamper_are_rejected(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    _rewrite_manifest(root, lambda manifest: manifest.__setitem__("unexpected", True))

    with pytest.raises(artifact.ArtifactValidationError, match="manifest keys"):
        artifact.verify_glm52_mtp_layer78(root)


def test_resigned_producer_commit_must_match_safetensors_metadata(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)

    def forge_producer_commit(manifest: dict[str, object]) -> None:
        producer = manifest["producer"]
        assert isinstance(producer, dict)
        producer["commit"] = "a" * 40

    _rewrite_manifest(root, forge_producer_commit)

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="producer commit/header metadata mismatch",
    ):
        artifact.verify_glm52_mtp_layer78(root, deep=True)


def test_output_and_tensor_digest_tamper_are_rejected(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE
    raw = bytearray(output.read_bytes())
    raw[-1] ^= 0xFF
    output.write_bytes(raw)

    with pytest.raises(
        artifact.ArtifactValidationError, match="artifact SHA-256|tensor SHA-256"
    ):
        artifact.verify_glm52_mtp_layer78(root)


def test_resigned_tamper_is_rejected_against_pinned_source_bytes(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE
    raw = bytearray(output.read_bytes())
    raw[-1] ^= 0xFF
    output.write_bytes(raw)

    def resign(manifest: dict[str, object]) -> None:
        artifact_row = manifest["artifact"]
        assert isinstance(artifact_row, dict)
        tensor_rows = artifact_row["tensors"]
        assert isinstance(tensor_rows, list)
        final_tensor = tensor_rows[-1]
        assert isinstance(final_tensor, dict)
        offsets = final_tensor["output_data_offsets"]
        assert isinstance(offsets, list)
        data_start = 8 + artifact_row["header_bytes"]
        final_tensor["sha256"] = hashlib.sha256(
            raw[data_start + offsets[0] : data_start + offsets[1]]
        ).hexdigest()
        artifact_row["sha256"] = hashlib.sha256(raw).hexdigest()

    _rewrite_manifest(root, resign)

    with pytest.raises(
        artifact.ArtifactValidationError, match="deep source tensor SHA-256"
    ):
        artifact.verify_glm52_mtp_layer78(root, deep=True)


def test_verified_handle_remains_bound_to_authenticated_inode_during_replacement(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE
    authenticated_bytes = output.read_bytes()
    replacement_bytes = bytearray(authenticated_bytes)
    replacement_bytes[-1] ^= 0xFF
    original_inspect_source = artifact._inspect_source
    replaced = False

    def replace_during_deep_verification(source_root: Path):
        nonlocal replaced
        if not replaced:
            replacement = root / ".replacement.safetensors"
            replacement.write_bytes(replacement_bytes)
            os.replace(replacement, output)
            replaced = True
        return original_inspect_source(source_root)

    monkeypatch.setattr(artifact, "_inspect_source", replace_during_deep_verification)

    with artifact.open_verified_glm52_mtp_layer78(root) as verified:
        assert replaced is True
        verified.file.seek(0)
        assert verified.file.read() == authenticated_bytes
        assert output.read_bytes() == replacement_bytes


def test_verified_handle_rejects_in_place_mutation_before_context_exit(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="artifact file changed while in use",
    ):
        with artifact.open_verified_glm52_mtp_layer78(root):
            with output.open("r+b") as mutable:
                mutable.seek(-1, os.SEEK_END)
                original = mutable.read(1)
                mutable.seek(-1, os.SEEK_END)
                mutable.write(bytes([original[0] ^ 0xFF]))
                mutable.flush()
                os.fsync(mutable.fileno())


def test_verified_handle_rejects_mutation_with_restored_mtime(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE
    original = output.stat()

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="artifact file changed while in use",
    ):
        with artifact.open_verified_glm52_mtp_layer78(root):
            with output.open("r+b") as mutable:
                mutable.seek(-1, os.SEEK_END)
                old = mutable.read(1)
                mutable.seek(-1, os.SEEK_END)
                mutable.write(bytes([old[0] ^ 0xFF]))
                mutable.flush()
                os.fsync(mutable.fileno())
            os.utime(
                output,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )


def test_verified_handle_checks_mutation_when_consumer_raises(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    output = root / artifact.ARTIFACT_FILE

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="artifact file changed while in use",
    ) as raised:
        with artifact.open_verified_glm52_mtp_layer78(root):
            with output.open("r+b") as mutable:
                mutable.seek(-1, os.SEEK_END)
                old = mutable.read(1)
                mutable.seek(-1, os.SEEK_END)
                mutable.write(bytes([old[0] ^ 0xFF]))
                mutable.flush()
                os.fsync(mutable.fileno())
            raise RuntimeError("synthetic consumer failure")

    assert isinstance(raised.value.__context__, RuntimeError)


def test_runtime_verifier_rejects_unauthenticated_shallow_mode(
    synthetic: SyntheticFixture,
) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="integrity-only and unauthenticated",
    ):
        artifact.verify_glm52_mtp_layer78(root, deep=False)


def test_missing_manifest_fails_closed(synthetic: SyntheticFixture) -> None:
    root = artifact.extract_glm52_mtp_layer78(synthetic.config)
    (root / artifact.MANIFEST_FILE).unlink()

    with pytest.raises(artifact.ArtifactValidationError, match="manifest"):
        artifact.verify_glm52_mtp_layer78(root)


def test_existing_destination_is_never_overwritten(synthetic: SyntheticFixture) -> None:
    synthetic.output.mkdir()
    sentinel = synthetic.output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(artifact.ArtifactPublicationError, match="already exists"):
        artifact.extract_glm52_mtp_layer78(synthetic.config)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_racing_destination_is_never_overwritten(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = artifact._rename_directory_exclusive

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "racer").write_text("keep", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(artifact, "_rename_directory_exclusive", race)

    with pytest.raises(artifact.ArtifactPublicationError, match="already exists"):
        artifact.extract_glm52_mtp_layer78(synthetic.config)
    assert (synthetic.output / "racer").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "case", ["symlink", "hardlink", "non_regular", "unsafe_mode", "unsafe_owner"]
)
def test_adversarial_source_file_is_rejected(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    target = synthetic.source / synthetic.shard_names[0]
    if case == "symlink":
        real = target.with_suffix(".real")
        target.rename(real)
        target.symlink_to(real.name)
    elif case == "hardlink":
        other = target.with_suffix(".hardlink")
        os.link(target, other)
    elif case == "non_regular":
        target.unlink()
        target.mkdir()
    elif case == "unsafe_mode":
        target.chmod(0o666)
    else:
        monkeypatch.setattr(artifact.os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(
        artifact.ArtifactValidationError,
        match="symlink|hardlink|regular|mode|owner|ownership",
    ):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


def test_dirty_producer_is_rejected(synthetic: SyntheticFixture) -> None:
    (synthetic.producer / "dirty.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(artifact.ArtifactValidationError, match="producer.*dirty"):
        artifact.preflight_glm52_mtp_layer78(synthetic.config)


@pytest.mark.parametrize("command", ["preflight", "extract", "verify"])
def test_cli_modes_have_help(command: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
    if command != "verify":
        assert "--source-root" in result.stdout
        assert "--producer-root" in result.stdout
    else:
        assert "--deep" in result.stdout


def test_verify_cli_has_no_unauthenticated_no_deep_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--output-root",
            str(tmp_path / "missing"),
            "--no-deep",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --no-deep" in result.stderr
