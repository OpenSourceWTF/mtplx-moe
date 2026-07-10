from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_cli import (
    add_expert_streaming_args,
    append_expert_streaming_child_args,
    expert_streaming_load_kwargs,
)
from mtplx.expert_streaming import RoutingPhase
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_expert_streaming_args(parser)
    return parser


def _model_root(tmp_path: Path, model_type: str = "hy_v3") -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    (root / "expert-manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_expert_cli_builds_explicit_bounded_config(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    args = _parser().parse_args(
        [
            "--expert-streaming",
            "--expert-memory-limit",
            "96GiB",
            "--expert-max-live-kv-tokens",
            "8192",
            "--expert-cache-limit",
            "24GiB",
            "--expert-runtime-reserve",
            "8GiB",
            "--no-expert-prefer-sidecar",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    config = kwargs["expert_streaming_config"]

    assert kwargs["mtp"] is False
    assert kwargs["expert_manifest"] == root / "expert-manifest.json"
    assert config.model_key == "hy3-q4"
    assert config.memory_limit_bytes == 96 * 1024**3
    assert config.max_live_kv_tokens == 8192
    assert config.expert_cache_limit_bytes == 24 * 1024**3
    assert config.runtime_reserve_bytes == 8 * 1024**3
    assert config.prefer_sidecar is False


def test_expert_cli_json_and_flags_are_strict_and_forwarded(tmp_path: Path) -> None:
    root = _model_root(tmp_path, "glm_moe_dsa")
    config_path = tmp_path / "stream.json"
    config_path.write_text(
        json.dumps(
            {
                "model_key": "glm52-q4",
                "memory_limit_bytes": "256GiB",
                "max_live_kv_tokens": 4096,
                "runtime_reserve_bytes": "12GiB",
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--expert-streaming-config",
            str(config_path),
            "--expert-memory-limit",
            "320GiB",
            "--no-expert-verify-record-hashes",
        ]
    )

    kwargs = expert_streaming_load_kwargs(args, root)
    assert kwargs["expert_streaming_config"].memory_limit_bytes == 320 * 1024**3
    assert kwargs["expert_streaming_config"].verify_record_hashes is False

    command = ["python", "-m", "mtplx.server.openai"]
    append_expert_streaming_child_args(command, args)
    assert "--expert-streaming" in command
    assert command[command.index("--expert-memory-limit") + 1] == "320GiB"
    assert "--no-expert-verify-record-hashes" in command


def test_expert_cli_requires_memory_and_kv_limits(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    args = _parser().parse_args(["--expert-streaming"])
    with pytest.raises(ValueError, match="missing memory-limit-bytes, max-live-kv-tokens"):
        expert_streaming_load_kwargs(args, root)


class _PhaseModel:
    def __init__(self) -> None:
        self.phases: list[RoutingPhase] = []

    def __call__(self, input_ids, cache=None):
        del cache
        self.phases.append(current_expert_routing_phase(token_count=999))
        return input_ids


class _StreamingStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self, *, timeout=None) -> None:
        del timeout
        self.closed = True

    def snapshot(self):
        return {"ok": True}

    def admit_kv_tokens(self, tokens):
        return SimpleNamespace(tokens=tokens)


def test_mtplx_runtime_marks_prefill_decode_and_closes_streaming_runtime() -> None:
    model = _PhaseModel()
    streaming = _StreamingStub()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=Path("model"),
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    runtime.forward_ar(SimpleNamespace(shape=(1, 3)))
    runtime.forward_ar(SimpleNamespace(shape=(1, 1)))

    assert model.phases == [RoutingPhase.PREFILL, RoutingPhase.DECODE]
    assert runtime.expert_streaming_snapshot() == {"ok": True}
    runtime.close()
    assert streaming.closed is True
