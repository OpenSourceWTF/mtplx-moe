"""Request resolution in benchmark_q2_mtp_depth_matrix must be per-entry.

Regression: the original dispatch was a binary ``if model == "hy3-q2" ... else
glm52`` — any campaign key added to the script's MODEL_SPECS (e.g.
``hy3-oq2e``) silently inherited GLM's model_root/manifest/mtp_artifacts and
failed at runtime with a crossed artifact path
(``glm52-mtp-layer78/layer80-bf16.safetensors``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("_depth_matrix_under_test", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oq2e_request_uses_its_own_entry_not_glm52():
    module = _load_runner()
    args = module.build_parser().parse_args(
        ["--model", "hy3-oq2e", "--hy3-depths", "2,3"]
    )
    (request,) = module._requests_from_args(args)
    assert request["model"] == "hy3-oq2e"
    assert "hy3-oq2e-mlx" in str(request["model_root"])
    assert "glm52" not in str(request["model_root"])
    assert "glm52" not in str(request["mtp_artifacts"])
    assert "hy3-bf16-and-mtp-layer80" in str(request["mtp_artifacts"])
    assert request["manifest"] == module._expand(
        Path(request["model_root"]) / "expert-manifest.json"
    )
    assert request["depths"] == (2, 3)


def test_known_keys_still_resolve_their_flag_overrides():
    module = _load_runner()
    args = module.build_parser().parse_args(
        [
            "--model", "hy3-q2",
            "--model", "glm52-q2",
            "--hy3-q2-model-root", "/tmp/custom-hy3-root",
            "--glm52-q2-mtp-artifacts", "/tmp/custom-glm-mtp",
        ]
    )
    hy3, glm = module._requests_from_args(args)
    assert str(hy3["model_root"]).endswith("custom-hy3-root")
    assert "hy3-bf16-and-mtp-layer80" in str(hy3["mtp_artifacts"])
    assert str(glm["mtp_artifacts"]).endswith("custom-glm-mtp")
    assert "glm52-expert-only-mlx-q2" in str(glm["model_root"])
