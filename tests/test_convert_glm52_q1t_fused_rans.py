from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_converter_binds_authoritative_q1_manifest_and_distinct_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.convert_glm52_q1t_fused_rans as converter

    source_root = tmp_path / "q1t"
    source_root.mkdir()
    q1 = object()
    authoritative = SimpleNamespace(
        model_key="glm52-expert-q1t",
        quant_mode="t158",
        quant_group_size=64,
        manifest_sha256="b" * 64,
        records=(
            SimpleNamespace(layer=3, expert=0),
            SimpleNamespace(layer=4, expert=0),
        ),
    )
    captured = {}

    monkeypatch.setattr(converter, "load_q1_manifest", lambda path: q1)
    monkeypatch.setattr(converter, "load_expert_manifest", lambda path: authoritative)

    def write(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return SimpleNamespace(file_bytes=1024, file_sha256="c" * 64)

    monkeypatch.setattr(converter, "write_glm52_q1t_fused_rans_artifact", write)

    assert converter.main(["--source-root", str(source_root), "--resume"]) == 0
    assert captured["source"] is q1
    assert captured["layers"] == (3, 4)
    assert captured["expected_expert_count"] == 1
    assert captured["source_expert_manifest_sha256"] == "b" * 64
    assert captured["resume"] is True
    assert captured["uniform_packed"] is False
    assert captured["output_bin"] == source_root / "experts-glm52-q1t-fused-rans.bin"
    assert captured["output_manifest"] == (
        source_root / "expert-manifest-glm52-q1t-fused-rans.json"
    )


def test_uniform_packed_converter_uses_distinct_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.convert_glm52_q1t_fused_rans as converter

    source_root = tmp_path / "q1t"
    source_root.mkdir()
    captured = {}
    authoritative = SimpleNamespace(
        model_key="glm52-expert-q1t",
        quant_mode="t158",
        quant_group_size=64,
        manifest_sha256="b" * 64,
        records=(SimpleNamespace(layer=3, expert=0),),
    )
    monkeypatch.setattr(converter, "load_q1_manifest", lambda _path: object())
    monkeypatch.setattr(
        converter,
        "load_expert_manifest",
        lambda _path: authoritative,
    )

    def write(_source, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(file_bytes=1024, file_sha256="c" * 64)

    monkeypatch.setattr(converter, "write_glm52_q1t_fused_rans_artifact", write)

    assert converter.main(
        ["--source-root", str(source_root), "--uniform-packed", "--layers", "3"]
    ) == 0
    assert captured["uniform_packed"] is True
    assert captured["output_bin"] == (
        source_root / "experts-glm52-q1t-fused-rans-uniform-packed.bin"
    )
    assert captured["output_manifest"] == (
        source_root
        / "expert-manifest-glm52-q1t-fused-rans-uniform-packed.json"
    )


def test_converter_rejects_non_q1t_authoritative_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    import scripts.convert_glm52_q1t_fused_rans as converter

    source_root = tmp_path / "q1t"
    source_root.mkdir()
    monkeypatch.setattr(converter, "load_q1_manifest", lambda path: object())
    monkeypatch.setattr(
        converter,
        "load_expert_manifest",
        lambda path: SimpleNamespace(
            model_key="glm52-expert-q2",
            quant_mode="affine",
            quant_group_size=64,
            manifest_sha256="b" * 64,
            records=(SimpleNamespace(layer=3, expert=0),),
        ),
    )

    with pytest.raises(ValueError, match="glm52-expert-q1t"):
        converter.main(["--source-root", str(source_root)])
