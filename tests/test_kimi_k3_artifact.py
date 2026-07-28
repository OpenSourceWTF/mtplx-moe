from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

import mtplx.expert_manifest as expert_manifest_module
from mtplx.expert_manifest import (
    ResidentTensor,
    load_expert_manifest,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import get_model_spec
from mtplx.kimi_k3_gguf import GGML_TYPE_BF16, GGML_TYPE_F32, GGUFTensor
import mtplx.kimi_k3_t158 as serializer
from mtplx.kimi_k3_t158 import (
    KIMI_K3_OFFICIAL_REVISION,
    KIMI_K3_SOURCE_REVISION,
    ConvertedLayer,
    KimiK3Layout,
    assemble_artifact,
    encode_expert_record,
)


def _official_fixture(root: Path) -> bytes:
    root.mkdir()
    config = {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "auto_map": {"AutoModel": "modeling_kimi_k3.KimiK3ForConditionalGeneration"},
        "text_config": {
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "hidden_act": "situ",
            "auto_map": {"AutoModel": "modeling_kimi_linear.KimiLinearForCausalLM"},
            "quantization_config": {"quant_method": "compressed-tensors"},
        },
    }
    original = json.dumps(config, indent=1).encode() + b"\n"
    (root / "config.json").write_bytes(original)
    files = {
        "tokenizer_config.json": b'{"auto_map":{"AutoTokenizer":["tokenization_kimi.TikTokenTokenizer",null]}}\n',
        "tiktoken.model": b"token bytes\n",
        "tokenization_kimi.py": b"from .encoding_k3 import EncodingK3\n",
        "encoding_k3.py": b"class EncodingK3: pass\n",
        "generation_config.json": b'{"eos_token_id":[163585,163586]}\n',
        "configuration_kimi_k3.py": b"class KimiK3Config: pass\n",
        "modeling_kimi_k3.py": b"class KimiK3ForConditionalGeneration: pass\n",
        "modeling_kimi_linear.py": b"class KimiLinearForCausalLM: pass\n",
        "kimi_k3_processor.py": b"class KimiK3Processor: pass\n",
        "kimi_k3_vision_processing.py": b"def process_vision(): pass\n",
        "media_utils.py": b"def load_media(): pass\n",
        "preprocessor_config.json": b'{"processor_class":"KimiK3Processor"}\n',
        "LICENSE": b"fixture license\n",
        "README.md": b"# fixture\n",
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    return original


class _TinyInventory:
    def __init__(self, root: Path, *, split_layer: bool = False) -> None:
        self.revision = KIMI_K3_SOURCE_REVISION
        self.layers = (1, 2)
        self.resident_descriptor_sha256 = "d" * 64
        self.files = tuple(
            SimpleNamespace(path=root / f"source-{index}.gguf", tensors=())
            for index in (1, 2, 3, 4)
        )
        tensors_by_file: list[list[GGUFTensor]] = [[], [], [], []]
        experts: list[GGUFTensor] = []
        for layer in (1, 2):
            owner = layer - 1
            for projection in ("gate", "up", "down"):
                tensor = GGUFTensor(
                    f"blk.{layer}.ffn_{projection}_exps.weight",
                    (256, 256, 2),
                    10,
                    0,
                )
                experts.append(tensor)
                tensor_owner = (
                    1 - owner
                    if split_layer and layer == 1 and projection == "up"
                    else owner
                )
                tensors_by_file[tensor_owner].append(tensor)
        text_a = GGUFTensor(
            "language_model.model.embed_tokens.weight",
            (1,),
            GGML_TYPE_F32,
            0,
        )
        vision = GGUFTensor(
            "vision_tower.patch.weight",
            (1,),
            GGML_TYPE_F32,
            0,
        )
        projector = GGUFTensor(
            "mm_projector.proj.weight",
            (1,),
            GGML_TYPE_F32,
            0,
        )
        text_b = GGUFTensor(
            "language_model.lm_head.weight",
            (2,),
            GGML_TYPE_BF16,
            0,
        )
        tensors_by_file[0].append(text_a)
        tensors_by_file[1].append(text_b)
        tensors_by_file[2].append(vision)
        tensors_by_file[3].append(projector)
        self.files = tuple(
            SimpleNamespace(path=source.path, tensors=tuple(tensors))
            for source, tensors in zip(self.files, tensors_by_file, strict=True)
        )
        self.expert_tensors = tuple(experts)
        self.resident_tensors = (text_a, text_b, projector, vision)

    def tensor_source(self, name: str):
        matches = [
            (source, tensor)
            for source in self.files
            for tensor in source.tensors
            if tensor.name == name
        ]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


@pytest.fixture
def tiny_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    source = tmp_path / "source"
    official = tmp_path / "official"
    output = tmp_path / "output"
    source.mkdir()
    original_config = _official_fixture(official)
    monkeypatch.setattr(
        serializer,
        "_OFFICIAL_FILE_SHA256",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in official.iterdir()
            if path.is_file()
        },
    )
    inventory = _TinyInventory(source)
    layout = KimiK3Layout(
        expert_count=2,
        layer_count=2,
        gate_shape=(256, 256),
        up_shape=(256, 256),
        down_shape=(256, 256),
    )

    monkeypatch.setattr(
        serializer,
        "inspect_kimi_k3_source",
        lambda root, revision: inventory,
    )
    monkeypatch.setattr(serializer, "KimiK3Layout", lambda: layout)

    source_indexes = {id(item): index for index, item in enumerate(inventory.files, 1)}

    def write_residents(
        output_path: Path,
        entries: tuple[tuple[str, str, tuple[int, ...], bytes], ...],
    ):
        header: dict[str, object] = {}
        cursor = 0
        for name, dtype, shape, payload in entries:
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + len(payload)],
            }
            cursor += len(payload)
        encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        prefix = len(encoded).to_bytes(8, "little") + encoded
        output_path.write_bytes(
            prefix + b"".join(payload for _name, _dtype, _shape, payload in entries)
        )
        residents: list[ResidentTensor] = []
        cursor = len(prefix)
        for name, dtype, shape, payload in entries:
            residents.append(
                ResidentTensor(
                    tensor=name,
                    shard=output_path.name,
                    offset=cursor,
                    length=len(payload),
                    dtype=dtype,
                    shape=shape,
                )
            )
            cursor += len(payload)
        return tuple(residents)

    def copy_residents(source_file, output_path: Path):
        entries_by_source = {
            1: (
                (
                    "language_model.model.embed_tokens.weight",
                    "F32",
                    (1,),
                    b"text",
                ),
            ),
            2: (("language_model.lm_head.weight", "BF16", (2,), b"head"),),
            3: (("vision_tower.patch.weight", "F32", (1,), b"visi"),),
            4: (("mm_projector.proj.weight", "F32", (1,), b"mmxx"),),
        }
        return write_residents(
            output_path,
            entries_by_source[source_indexes[id(source_file)]],
        )

    def convert(source_file, output_dir, _inventory, *, layer, resume, layout):
        assert source_file is inventory.files[layer - 1]
        path = output_dir / f"experts-t158-layer-{layer:03d}-of-002.bin"
        journal = path.with_name(path.name + ".journal.jsonl")
        records = []
        payload = bytearray()
        for expert in range(2):
            record = encode_expert_record(
                {
                    "gate_proj": np.full(
                        layout.gate_shape, layer + expert, dtype=np.float32
                    ),
                    "up_proj": np.full(
                        layout.up_shape, layer + expert + 1, dtype=np.float32
                    ),
                    "down_proj": np.full(
                        layout.down_shape, layer + expert + 2, dtype=np.float32
                    ),
                },
                layer=layer,
                expert=expert,
                shard=path.name,
                record_offset=len(payload),
            )
            assert record.payload is not None
            payload.extend(record.payload)
            records.append(record.metadata_only())
        path.write_bytes(payload)
        journal.write_text("{}\n", encoding="utf-8")
        return ConvertedLayer(
            layer=layer,
            path=path,
            journal_path=journal,
            logical_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            records=tuple(records),
        )

    monkeypatch.setattr(serializer, "copy_resident_safetensors", copy_residents)
    monkeypatch.setattr(serializer, "convert_layer", convert)
    routed_bytes = layout.layer_count * layout.expert_count * layout.record_bytes
    tiny_spec = replace(
        get_model_spec("kimi-k3-q1t"),
        total_tensor_bytes=8 + routed_bytes,
        total_layers=3,
        routed_layer_count=2,
        expert_count=2,
        top_k=1,
        hidden_size=256,
        model_hidden_size=8,
        expert_hidden_size=256,
        router_bytes=0,
        kv_bytes_per_token=0,
        fixed_cache_bytes_per_batch=0,
    )
    monkeypatch.setattr(
        expert_manifest_module,
        "get_model_spec",
        lambda key: tiny_spec if key == "kimi-k3-q1t" else get_model_spec(key),
    )
    return SimpleNamespace(
        source=source,
        official=official,
        output=output,
        inventory=inventory,
        layout=layout,
        original_config=original_config,
    )


def _assemble(fixture: SimpleNamespace, **overrides):
    arguments = {
        "source_revision": KIMI_K3_SOURCE_REVISION,
        "official_revision": KIMI_K3_OFFICIAL_REVISION,
        "official_root": fixture.official,
        "resume": False,
        "layers": None,
    }
    arguments.update(overrides)
    return assemble_artifact(
        fixture.source,
        fixture.output,
        **arguments,
    )


def test_partial_layer_selection_never_publishes_final_manifest(
    tiny_assembly: SimpleNamespace,
) -> None:
    result = _assemble(tiny_assembly, layers=(1,))

    assert result.complete is False
    assert result.converted_layers == (1,)
    assert result.manifest_path is None
    assert (tiny_assembly.output / "experts-t158-layer-001-of-002.bin").is_file()
    assert not (tiny_assembly.output / "expert-manifest.json").exists()
    assert not (tiny_assembly.output / "model.safetensors.index.json").exists()


def test_empty_selection_never_inspects_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        serializer,
        "inspect_kimi_k3_source",
        lambda *_args: pytest.fail("empty pilot must not inspect or write"),
    )
    result = assemble_artifact(
        tmp_path / "source",
        tmp_path / "output",
        source_revision=KIMI_K3_SOURCE_REVISION,
        official_revision=KIMI_K3_OFFICIAL_REVISION,
        official_root=tmp_path / "official",
        resume=True,
        layers=(),
    )

    assert result.complete is False
    assert result.converted_layers == ()
    assert not (tmp_path / "output").exists()


def test_full_assembly_writes_authoritative_multipart_artifact(
    tiny_assembly: SimpleNamespace,
) -> None:
    result = _assemble(tiny_assembly)

    assert result.complete is True
    assert result.converted_layers == (1, 2)
    assert result.manifest_path == tiny_assembly.output / "expert-manifest.json"
    manifest = load_expert_manifest(result.manifest_path)
    assert manifest.model_key == "kimi-k3-q1t"
    assert manifest.source_repo == "GrEarl/Kimi-K3-GGUF"
    assert manifest.quant_mode == "t158"
    assert manifest.quant_bits == 2
    assert manifest.quant_group_size == 64
    assert len(manifest.sidecar.parts) == 2
    assert len(manifest.records) == 4
    # The production manifest has 82,432 records and must stay below the
    # runtime's 128 MiB bounded JSON reader.
    assert result.manifest_path.read_bytes().count(b"\n") == 1
    assert all(len(record.segments) == 6 for record in manifest.records)
    assert [(record.layer, record.expert) for record in manifest.records] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    for record in manifest.records:
        part = manifest.sidecar.part_for(record)
        assert record.sidecar_offset is not None
        assert record.sidecar_length == record.logical_bytes
        assert record.sidecar_offset + record.logical_bytes <= part.size
        assert all(segment.shard == part.file for segment in record.segments)

    assert {tensor.tensor for tensor in manifest.resident_tensors} == {
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
    }
    shard_kinds = {shard.name: shard.kind for shard in manifest.shards}
    assert shard_kinds["resident-00001-of-00004.safetensors"] == "safetensors"
    assert shard_kinds["resident-00002-of-00004.safetensors"] == "safetensors"
    assert shard_kinds["resident-00003-of-00004.safetensors"] == "preserved-safetensors"
    assert shard_kinds["resident-00004-of-00004.safetensors"] == "preserved-safetensors"
    assert manifest.resident_tensor_bytes == 8
    assert manifest.routed_expert_bytes == 4 * tiny_assembly.layout.record_bytes
    assert manifest.artifact_tensor_bytes == 8 + 4 * tiny_assembly.layout.record_bytes

    index = json.loads(
        (tiny_assembly.output / "model.safetensors.index.json").read_bytes()
    )
    assert set(index["weight_map"]) == {
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "mm_projector.proj.weight",
        "vision_tower.patch.weight",
    }
    assert index["metadata"]["total_size"] == 16
    report = verify_expert_manifest(
        manifest,
        tiny_assembly.output,
        verify_records=True,
        verify_shard_hashes=True,
        verify_sidecar_hash=True,
    )
    assert report["valid"] is True


def test_assembly_flattens_text_config_and_copies_audited_runtime_exactly(
    tiny_assembly: SimpleNamespace,
) -> None:
    _assemble(tiny_assembly)

    config = json.loads((tiny_assembly.output / "config.json").read_bytes())
    assert config["model_type"] == "kimi_linear"
    assert config["hidden_act"] == "situ"
    assert "text_config" not in config
    assert "quantization_config" not in config
    assert "auto_map" not in config
    assert (
        tiny_assembly.output / "config.kimi_k3.original.json"
    ).read_bytes() == tiny_assembly.original_config
    for name in (
        "tokenizer_config.json",
        "tiktoken.model",
        "tokenization_kimi.py",
        "encoding_k3.py",
        "generation_config.json",
        "configuration_kimi_k3.py",
        "modeling_kimi_k3.py",
        "modeling_kimi_linear.py",
        "kimi_k3_processor.py",
        "kimi_k3_vision_processing.py",
        "media_utils.py",
        "preprocessor_config.json",
        "LICENSE",
        "README.md",
    ):
        assert (tiny_assembly.output / name).read_bytes() == (
            tiny_assembly.official / name
        ).read_bytes()
    original = json.loads(
        (tiny_assembly.output / "config.kimi_k3.original.json").read_bytes()
    )
    for target in original["auto_map"].values():
        module = target.split(".", 1)[0]
        assert (tiny_assembly.output / f"{module}.py").is_file()

    receipt = json.loads(
        (tiny_assembly.output / "conversion-receipt.json").read_bytes()
    )
    assert receipt["source"]["revision"] == KIMI_K3_SOURCE_REVISION
    assert receipt["official"]["revision"] == KIMI_K3_OFFICIAL_REVISION
    assert receipt["residents"]["preserved_tensor_bytes"] == 16
    assert receipt["residents"]["text_tensor_bytes"] == 8
    assert receipt["residents"]["non_text_tensor_bytes"] == 8
    assert receipt["codec"]["routed_tensor_bytes"] == (
        4 * tiny_assembly.layout.record_bytes
    )
    for name, item in receipt["official"]["files"].items():
        assert (
            item["sha256"]
            == hashlib.sha256((tiny_assembly.official / name).read_bytes()).hexdigest()
        )
        assert item["size"] == (tiny_assembly.official / name).stat().st_size


def test_assembly_refuses_split_projection_ownership(
    tiny_assembly: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _TinyInventory(tiny_assembly.source, split_layer=True)
    monkeypatch.setattr(
        serializer,
        "inspect_kimi_k3_source",
        lambda root, revision: inventory,
    )

    with pytest.raises(ValueError, match="same GGUF shard"):
        _assemble(tiny_assembly, layers=(1,))
    assert not (tiny_assembly.output / "expert-manifest.json").exists()


def test_assembly_rejects_official_metadata_not_bound_to_revision(
    tiny_assembly: SimpleNamespace,
) -> None:
    (tiny_assembly.official / "tokenization_kimi.py").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="pinned official revision"):
        _assemble(tiny_assembly)
    assert not (tiny_assembly.output / "expert-manifest.json").exists()


def test_assembly_rejects_missing_pinned_processor_model_bundle(
    tiny_assembly: SimpleNamespace,
) -> None:
    (tiny_assembly.official / "modeling_kimi_linear.py").unlink()

    with pytest.raises(ValueError, match="modeling_kimi_linear.py"):
        _assemble(tiny_assembly)
    assert not (tiny_assembly.output / "expert-manifest.json").exists()


def test_assembly_no_overwrite_and_resume_validates_existing_metadata(
    tiny_assembly: SimpleNamespace,
) -> None:
    first = _assemble(tiny_assembly)
    with pytest.raises(ValueError, match="overwrite|exists"):
        _assemble(tiny_assembly)

    resumed = _assemble(tiny_assembly, resume=True)
    assert resumed.manifest_path == first.manifest_path

    (tiny_assembly.output / "config.json").write_text('{"tampered":true}\n')
    with pytest.raises(ValueError, match="config.json"):
        _assemble(tiny_assembly, resume=True)


@pytest.mark.parametrize(
    "name",
    (
        "config.json",
        "config.kimi_k3.original.json",
        *serializer._AUDITED_RUNTIME_FILES,
        *serializer._OPTIONAL_DOCUMENTATION_FILES,
        "model.safetensors.index.json",
        "conversion-receipt.json",
    ),
)
def test_exact_installer_resumes_owned_metadata_prefix(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / name
    payload = (name.encode() + b"\0") * 257
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload[: len(payload) // 3])

    serializer._install_bytes_exact(path, payload, resume=True)

    assert path.read_bytes() == payload
    assert not partial.exists()


def test_exact_installer_resumes_large_manifest_prefix_chunkwise(
    tmp_path: Path,
) -> None:
    path = tmp_path / "expert-manifest.json"
    payload = b"m" * (serializer._COPY_CHUNK_BYTES + 4097)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload[: serializer._COPY_CHUNK_BYTES + 17])

    serializer._install_bytes_exact(path, payload, resume=True)

    assert path.stat().st_size == len(payload)
    assert (
        hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(payload).digest()
    )


@pytest.mark.parametrize(
    "failure",
    ("longer", "mismatch", "symlink", "directory", "hardlink"),
)
def test_exact_installer_rejects_unowned_or_invalid_partial(
    tmp_path: Path,
    failure: str,
) -> None:
    path = tmp_path / "model.safetensors.index.json"
    payload = b"authoritative metadata payload"
    partial = path.with_name(path.name + ".partial")
    if failure == "longer":
        partial.write_bytes(payload + b"x")
    elif failure == "mismatch":
        partial.write_bytes(b"wrong")
    elif failure == "symlink":
        target = tmp_path / "elsewhere"
        target.write_bytes(payload[:3])
        partial.symlink_to(target)
    elif failure == "hardlink":
        target = tmp_path / "elsewhere"
        target.write_bytes(payload[:3])
        os.link(target, partial)
    else:
        partial.mkdir()

    with pytest.raises(ValueError, match="partial|match"):
        serializer._install_bytes_exact(path, payload, resume=True)
    assert not path.exists()


def test_exact_installer_does_not_resume_prefix_without_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversion-receipt.json"
    payload = b"receipt payload"
    path.with_name(path.name + ".partial").write_bytes(payload[:4])

    with pytest.raises(ValueError, match="partial"):
        serializer._install_bytes_exact(path, payload, resume=False)
    assert not path.exists()


def test_assembly_recovers_manifest_publication_prefix_last(
    tiny_assembly: SimpleNamespace,
) -> None:
    first = _assemble(tiny_assembly)
    assert first.manifest_path is not None
    payload = first.manifest_path.read_bytes()
    first.manifest_path.unlink()
    partial = first.manifest_path.with_name(first.manifest_path.name + ".partial")
    partial.write_bytes(payload[: len(payload) // 2])

    resumed = _assemble(tiny_assembly, resume=True)

    assert resumed.complete is True
    assert resumed.manifest_path is not None
    assert resumed.manifest_path.read_bytes() == payload
    assert not partial.exists()


def test_assembly_verifies_physical_inventory_before_publication(
    tiny_assembly: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    real_verify = verify_expert_manifest

    def verify_before_publish(manifest, root, **kwargs):
        root = Path(root)
        assert not (root / "expert-manifest.json").exists()
        assert (root / "model.safetensors.index.json").is_file()
        calls.append(root)
        return real_verify(manifest, root, **kwargs)

    monkeypatch.setattr(
        serializer,
        "verify_expert_manifest",
        verify_before_publish,
    )

    result = _assemble(tiny_assembly)

    assert result.complete is True
    assert calls == [tiny_assembly.output]


@pytest.mark.parametrize(
    ("source_revision", "official_revision"),
    [
        ("main", KIMI_K3_OFFICIAL_REVISION),
        (KIMI_K3_SOURCE_REVISION, "main"),
    ],
)
def test_assembly_refuses_unpinned_revisions(
    tmp_path: Path,
    source_revision: str,
    official_revision: str,
) -> None:
    with pytest.raises(ValueError, match="pinned"):
        assemble_artifact(
            tmp_path / "source",
            tmp_path / "output",
            source_revision=source_revision,
            official_revision=official_revision,
            official_root=tmp_path / "official",
            resume=True,
            layers=(),
        )


def test_converter_cli_exposes_pinned_inspection_json() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "convert_kimi_k3_q2k_t158.py"
    )
    source = script.read_text(encoding="utf-8")

    assert "--source-revision" in source
    assert "--official-revision" in source
    assert "--official-metadata" in source
    assert "--inspect" in source
    assert "--json" in source


def test_converter_cli_layer_parser_accepts_ranges_and_rejects_duplicates() -> None:
    from scripts.convert_kimi_k3_q2k_t158 import parse_layers

    assert parse_layers("1,3-5,92") == (1, 3, 4, 5, 92)
    with pytest.raises(ValueError, match="duplicate"):
        parse_layers("1,1")
    with pytest.raises(ValueError, match="1..92"):
        parse_layers("0")


def test_converter_cli_inspect_json_never_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.convert_kimi_k3_q2k_t158 as cli

    calls: list[str] = []

    class Projection:
        def to_dict(self):
            return {"source_tensor_bytes": 100, "output_tensor_bytes": 75}

    def project(*_args, **_kwargs):
        calls.append("inspect")
        return Projection()

    monkeypatch.setattr(cli, "project_artifact", project)
    monkeypatch.setattr(
        cli,
        "assemble_artifact",
        lambda *_args, **_kwargs: pytest.fail("inspect must not write"),
    )
    exit_code = cli.main(
        [
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "output"),
            "--source-revision",
            KIMI_K3_SOURCE_REVISION,
            "--official-revision",
            KIMI_K3_OFFICIAL_REVISION,
            "--official-metadata",
            str(tmp_path / "official"),
            "--inspect",
            "--json",
        ]
    )

    assert exit_code == 0
    assert calls == ["inspect"]
    assert json.loads(capsys.readouterr().out) == {
        "output_tensor_bytes": 75,
        "source_tensor_bytes": 100,
    }
