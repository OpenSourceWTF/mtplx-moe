from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "summarize_issue51_next_k.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_issue51_next_k", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(context: int) -> dict[str, object]:
    return {
        "token_count": context,
        "token_sha256": f"token-{context}",
        "prompt_policy": "realistic_programming_v1",
        "prompt_format": "chat",
        "prompt_release_valid": True,
        "prompt_tail_sha256": "a" * 64,
        "prompt_filler_sha256": f"filler-{context}",
        "prompt_artifact_kinds": 6,
    }


def _row(context: int, depth: int, *, diagnostic: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "context_tokens": context,
        "requested_depth": depth,
        "prompt_identity": _identity(context),
        "prompt_target_prefill_time_s": 4.0,
        "decode_elapsed_s": 8.0 if depth == 0 else 6.0,
        "decode_tok_s": 16.0 if depth == 0 else 20.0,
        "decode_expert_cache_hit_rate": 0.9,
        "accepted_drafts": 0 if depth == 0 else 80,
        "evaluated_drafts": 0 if depth == 0 else 100,
        "conditional_hit_rate": None if depth == 0 else 0.8,
        "verify_calls": 128 if depth == 0 else 64,
        "ar_comparison": {"status": "reference" if depth == 0 else "exact"},
        "resource_telemetry": None,
    }
    if diagnostic:
        row["resource_telemetry"] = {
            "schema": "mtplx-resource-telemetry-v2",
            "memory": {
                "peak_memory_bytes": 3 * 1024**3,
                "limit_bytes": 112 * 1024**3,
                "utilization_of_limit": 3 / 112,
            },
            "host": {"generation_thread_core_fraction": 0.75},
            "storage": {
                "mean_gib_per_second": 2.0,
                "utilization_of_ceiling": 0.16,
            },
            "reader_pool": {
                "mean_active_readers": 1.25,
                "lifetime_active_readers_peak": 8,
            },
            "powermetrics": {
                "available": True,
                "process_cpu_ms_per_s_mean": 920.0,
                "process_gpu_busy_fraction": 0.66,
            },
            "coverage": {"gpu": "measured_process_time"},
        }
    return row


def _payload(*, diagnostic: bool) -> dict[str, object]:
    lane = (
        "diagnostic-resource-instrumented"
        if diagnostic
        else "headline-uninstrumented"
    )
    return {
        "schema": "mtplx-q2-bf16-mtp-depth-matrix-v3",
        "status": "passed",
        "passed": True,
        "configuration": {
            "measurement_lane": lane,
            "contexts": [1024],
            "output_tokens": 1028,
        },
        "models": [
            {
                "model": "hy3-q2",
                "model_key": "hy3-expert-q2",
                "measurement_lane": lane,
                "prompts": [
                    {
                        "context_tokens": 1024,
                        **_identity(1024),
                        "builder_metadata": _identity(1024),
                    }
                ],
                "observations": [
                    _row(1024, 0, diagnostic=diagnostic),
                    _row(1024, 1, diagnostic=diagnostic),
                ],
            }
        ],
    }


def test_summary_requires_matched_prompts_and_complementary_lanes() -> None:
    module = _load_module()
    headline = _payload(diagnostic=False)
    diagnostic = _payload(diagnostic=True)

    summary = module.summarize(headline, diagnostic)

    assert summary["prompt_policy"] == "realistic_programming_v1"
    assert summary["prompt_format"] == "chat"
    assert len(summary["rows"]) == 2

    wrong_prompt = copy.deepcopy(diagnostic)
    wrong_prompt["models"][0]["observations"][0]["prompt_identity"][
        "token_sha256"
    ] = "different"
    with pytest.raises(ValueError, match="prompt identity"):
        module.summarize(headline, wrong_prompt)

    wrong_lane = copy.deepcopy(diagnostic)
    wrong_lane["configuration"]["measurement_lane"] = "headline-uninstrumented"
    with pytest.raises(ValueError, match="diagnostic-resource-instrumented"):
        module.summarize(headline, wrong_lane)


def test_big_table_renders_headline_speed_and_resource_utilization() -> None:
    module = _load_module()
    markdown = module.render_markdown(
        module.summarize(_payload(diagnostic=False), _payload(diagnostic=True))
    )

    assert "realistic_programming_v1" in markdown
    assert "headline telemetry-off" in markdown
    assert "matched telemetry-on" in markdown
    for heading in ("Memory GiB / %", "CPU cores p/g", "SSD GiB/s / %", "GPU busy"):
        assert heading in markdown
    assert "3.00 / 2.7%" in markdown
    assert "0.92 / 0.75" in markdown
    assert "2.00 / 16.0%" in markdown
    assert "66.0%" in markdown
    assert "20.000" in markdown
    assert "+25.0%" in markdown
