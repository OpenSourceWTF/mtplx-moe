from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "compare_issue64_draft_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_issue64_draft_artifacts",
        _SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(token_sha256: str = "same-prompt") -> dict[str, object]:
    return {
        "token_count": 1024,
        "token_sha256": token_sha256,
        "prompt_policy": "realistic_programming_v1",
        "prompt_format": "chat",
        "prompt_release_valid": True,
        "prompt_tail_sha256": "tail",
        "prompt_filler_sha256": "filler",
        "prompt_artifact_kinds": 6,
    }


def _row(depth: int, *, core: str) -> dict[str, object]:
    return {
        "context_tokens": 1024,
        "requested_depth": depth,
        "replicate": 1,
        "prompt_identity": _identity(),
        "decode_tok_s": 6.0 if core == "stock" else 6.6,
        "token_ids": [depth, 10, 11],
        "accepted_drafts": 0 if depth == 0 else 7,
        "drafted_tokens": 0 if depth == 0 else 9,
        "evaluated_drafts": 0 if depth == 0 else 8,
        "acceptance_by_depth": [] if depth == 0 else [0.875],
        "verify_calls": 3,
        "final_state_contract": {"safe_to_commit": True, "mtp_cache_offset": 12},
        "speculative_event_contract": {"passed": True},
    }


def _payload(core: str) -> dict[str, object]:
    return {
        "schema": "mtplx-q2-bf16-mtp-depth-matrix-v3",
        "status": "passed",
        "passed": True,
        "configuration": {
            "contexts": [1024],
            "output_tokens": 1028,
            "sampler": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
            "generation": {
                "draft_core": core,
                "verify_strategy": "capture_commit",
                "mtp_cache_policy": "persistent",
                "mtp_history_policy": "committed",
            },
            "candidate": {
                "draft_core": core,
                "verify_strategy": "capture_commit",
                "compiled_verify_mode": "off",
            },
        },
        "models": [
            {
                "model": "hy3-q2",
                "model_key": "hy3-expert-q2",
                "observations": [_row(0, core=core), _row(3, core=core)],
            }
        ],
    }


def test_comparison_refuses_prompt_mismatch_before_performance_delta() -> None:
    module = _load_module()
    stock = _payload("stock")
    device = _payload("device-k")
    device["models"][0]["observations"][1]["prompt_identity"] = _identity(
        "different-prompt"
    )
    device["models"][0]["observations"][1]["decode_tok_s"] = "not-a-number"
    device["models"][0]["observations"].append(_row(7, core="device-k"))

    with pytest.raises(
        module.ArtifactComparisonError,
        match=(
            r"prompt identity mismatch for context=1024 depth=3 replicate=1: "
            r"stock token_sha256=same-prompt, device-k "
            r"token_sha256=different-prompt; performance comparison refused"
        ),
    ):
        module.compare_artifacts(stock, device)


def test_comparison_calculates_deltas_only_for_same_prompt_and_configuration() -> None:
    module = _load_module()

    report = module.compare_artifacts(_payload("stock"), _payload("device-k"))

    assert report["status"] == "comparable"
    assert report["prompt_identity_match"] is True
    assert report["stock_draft_core"] == "stock"
    assert report["device_draft_core"] == "device-k"
    assert [row["device_over_stock_ratio"] for row in report["rows"]] == [1.1, 1.1]
    assert all(row["tokens_identical"] for row in report["rows"])
    assert all(row["acceptance_identical"] for row in report["rows"])
    assert all(row["final_state_identical"] for row in report["rows"])

    wrong_sampler = copy.deepcopy(_payload("device-k"))
    wrong_sampler["configuration"]["sampler"]["seed"] = 9
    with pytest.raises(module.ArtifactComparisonError, match="configuration sampler"):
        module.compare_artifacts(_payload("stock"), wrong_sampler)


def test_comparison_requires_stock_and_device_k_arms() -> None:
    module = _load_module()

    with pytest.raises(module.ArtifactComparisonError, match="stock arm draft_core"):
        module.compare_artifacts(_payload("device-k"), _payload("device-k"))


def test_comparison_cli_emits_machine_readable_report(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_module()
    stock_path = tmp_path / "stock.json"
    device_path = tmp_path / "device.json"
    output_path = tmp_path / "comparison.json"
    stock_path.write_text(json.dumps(_payload("stock")), encoding="utf-8")
    device_path.write_text(json.dumps(_payload("device-k")), encoding="utf-8")

    assert (
        module.main(
            [
                "--stock-json",
                str(stock_path),
                "--device-json",
                str(device_path),
                "--output-json",
                str(output_path),
            ]
        )
        == 0
    )

    stdout_report = json.loads(capsys.readouterr().out)
    stored_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_report == stored_report
    assert stored_report["status"] == "comparable"
