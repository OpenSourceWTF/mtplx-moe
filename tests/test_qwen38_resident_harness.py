from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _harness():
    path = Path(__file__).parents[1] / "scripts/qwen38_flash_next_oq4_harness.py"
    spec = importlib.util.spec_from_file_location("qwen38_resident_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_loads_resident_model_and_only_configures_ngram_cache() -> None:
    harness = _harness()
    args = SimpleNamespace(
        mode="mtp",
        prompt_tokens=32,
        max_tokens=8,
        ngram_cache_gib=1,
        runtime_target_gib=75,
    )

    kwargs = harness._resident_load_kwargs(args)

    assert kwargs == {
        "mtp": True,
        "ngram_cache_limit_bytes": 1024**3,
        "ngram_context_tokens": 40,
        "ngram_target_residency_bytes": 75 * 1024**3,
    }
    assert not any("expert" in key for key in kwargs)


def test_harness_defaults_to_smallest_viable_resident_target() -> None:
    harness = _harness()

    args = harness._parse_args(["--smoke", "--preflight-only"])

    assert args.runtime_target_gib == 82
