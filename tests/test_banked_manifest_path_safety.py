"""The banked manifest must validate its bin filename like every sibling.

`bin_path()` does `self.path.parent / self.file`, so an absolute string
replaces the base entirely and a `../` component escapes the artifact root.
This is dormant while banks are built locally and becomes live the moment a
banked artifact is downloaded from the Hub.
"""

from __future__ import annotations

import json

import pytest

from mtplx.expert_banked import BankedManifestError, load_banked_manifest


def _manifest(file_value: str) -> dict:
    """Minimal but schema-valid banked manifest.

    The parser requires component length == prod(shape) * itemsize *
    expert_count, and the layer length to equal the sum of its components.
    With expert_count=1 and a [4096, 1] U32 component that is 16384 bytes --
    exactly one alignment unit.
    """

    return {
        "format": "mtplx-banked-expert-banks-v1",
        "model_key": "hy3-expert-q2",
        "file": file_value,
        "codec": "none",
        "alignment": 16384,
        "expert_count": 1,
        "layers": [
            {
                "layer": 1,
                "offset": 0,
                "length": 16384,
                "sha256": "0" * 64,
                "components": [
                    {
                        "component": "gate_proj.weight",
                        "dtype": "U32",
                        "shape": [4096, 1],
                        "offset": 0,
                        "length": 16384,
                    }
                ],
            }
        ],
    }


def write(tmp_path, file_value: str):
    path = tmp_path / "experts-banked-manifest.json"
    path.write_text(json.dumps(_manifest(file_value)))
    return path


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "../../../etc/passwd",
        "../sibling-artifact/experts.bin",
        "sub\\dir\\experts.bin",
    ],
)
def test_hostile_bin_names_are_rejected(tmp_path, hostile) -> None:
    with pytest.raises((BankedManifestError, ValueError)):
        load_banked_manifest(write(tmp_path, hostile))


@pytest.mark.parametrize(
    "benign",
    [
        "experts-banked.bin",
        "shards/experts-00001-of-00006.bin",
        # PurePosixPath normalises "./" away, so this is the plain name --
        # benign, and worth pinning so nobody "hardens" it into a rejection.
        "./experts-banked.bin",
    ],
)
def test_relative_names_including_subdirectories_are_accepted(tmp_path, benign) -> None:
    manifest = load_banked_manifest(write(tmp_path, benign))
    # The resolved bin must stay inside the artifact root; exact string
    # equality is not the contract (normalisation may rewrite it).
    resolved = manifest.bin_path().resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))
