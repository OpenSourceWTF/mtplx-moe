"""Teacher-forced MTP draft-rank probe: CLI contract and tiny-fixture runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest

import test_hy3_streamed_mtp as streamed_fixtures

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "probe_mtp_draft_rank.py"
)

_BASE_ARGS = [
    "/model",
    "/manifest",
    "--model-key",
    "hy3-q4",
    "--memory-limit",
    "112GiB",
    "--max-live-kv-tokens",
    "2048",
    "--prompt-file",
    "/prompt.md",
]

RECORD_KEYS = {
    "position",
    "input_token",
    "true_token",
    "rank",
    "true_prob",
    "argmax_token",
    "argmax_prob",
    "echo",
}
SUMMARY_KEYS = {
    "positions",
    "acceptance_at_1",
    "top5_rate",
    "top20_rate",
    "median_rank",
    "echo_rate",
    "mean_true_prob",
}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_mtp_draft_rank", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_pin_the_probe_contract() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(_BASE_ARGS)
    assert args.probe_positions == 200
    assert args.prefill_chunk == 256
    assert args.draft_batch == 32
    assert args.output_json is None
    assert args.chat is False
    assert args.enable_mtp is False
    assert args.mtp_artifacts is None
    assert args.mtp_precision is None
    assert str(args.prompt_file) == "/prompt.md"


def test_prompt_file_is_required(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_BASE_ARGS[:-2])
    assert "--prompt-file" in capsys.readouterr().err


def test_validate_requires_enable_mtp_and_artifacts(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()

    # The probe is meaningless without the head: fail closed, not degrade.
    args = parser.parse_args(_BASE_ARGS)
    with pytest.raises(SystemExit):
        module.validate_probe_flags(parser, args)
    assert "--enable-mtp" in capsys.readouterr().err

    args = parser.parse_args([*_BASE_ARGS, "--enable-mtp"])
    with pytest.raises(SystemExit):
        module.validate_probe_flags(parser, args)
    assert "--mtp-artifacts" in capsys.readouterr().err

    hy3_free = [arg for arg in _BASE_ARGS if arg != "hy3-q4"]
    hy3_free.insert(_BASE_ARGS.index("hy3-q4"), "glm52-q4")
    args = parser.parse_args(
        [*hy3_free, "--enable-mtp", "--mtp-artifacts", "/artifacts"]
    )
    with pytest.raises(SystemExit):
        module.validate_probe_flags(parser, args)
    assert "hy3-q4" in capsys.readouterr().err


def test_validate_defaults_mtp_precision_to_bf16() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [*_BASE_ARGS, "--enable-mtp", "--mtp-artifacts", "/artifacts"]
    )
    module.validate_probe_flags(parser, args)
    assert args.mtp_precision == "bf16"

    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
            "--mtp-precision",
            "q4",
        ]
    )
    module.validate_probe_flags(parser, args)
    assert args.mtp_precision == "q4"


def test_probe_rejects_runtimes_without_mtp(tmp_path: Path) -> None:
    module = _load_module()
    runtime, rt_ar, _rt_mtp = streamed_fixtures._injected_runtime_pair(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="MTP-enabled"):
            module.probe_draft_ranks(rt_ar, [1, 2, 3, 4], probe_positions=2)
    finally:
        runtime.close()


PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_tiny_fixture_probe_schema_ranks_and_summary(tmp_path: Path) -> None:
    module = _load_module()
    runtime, _rt_ar, rt_mtp = streamed_fixtures._injected_runtime_pair(tmp_path)
    try:
        probe = module.probe_draft_ranks(rt_mtp, PROMPT, probe_positions=6)
    finally:
        runtime.close()

    assert probe["prompt_tokens"] == len(PROMPT)
    assert probe["probe_positions"] == 6
    assert probe["probe_positions_capped"] is False
    assert set(probe["variants"]) == {"post_norm", "pre_norm"}

    vocab = 128
    for payload in probe["variants"].values():
        records = payload["records"]
        # The last 6 usable positions of a 12-token prompt: hiddens 4..9,
        # head inputs t_5..t_10, targets t_6..t_11.
        assert [record["position"] for record in records] == list(range(4, 10))
        for record in records:
            assert set(record) == RECORD_KEYS
            position = record["position"]
            assert record["input_token"] == PROMPT[position + 1]
            assert record["true_token"] == PROMPT[position + 2]
            assert isinstance(record["rank"], int)
            assert 0 <= record["rank"] < vocab
            assert 0.0 <= record["true_prob"] <= 1.0
            assert 0.0 <= record["argmax_prob"] <= 1.0
            assert record["argmax_prob"] >= record["true_prob"] - 1e-6
            assert record["echo"] == (
                record["argmax_token"] == record["input_token"]
            )
            # rank 0 means the true token IS the argmax.
            assert (record["rank"] == 0) == (
                record["argmax_token"] == record["true_token"]
            )
        summary = payload["summary"]
        assert set(summary) == SUMMARY_KEYS
        assert summary["positions"] == len(records)
        ranks = [record["rank"] for record in records]
        assert summary["acceptance_at_1"] == pytest.approx(
            sum(rank == 0 for rank in ranks) / len(ranks)
        )
        assert summary["top5_rate"] == pytest.approx(
            sum(rank < 5 for rank in ranks) / len(ranks)
        )
        assert summary["echo_rate"] == pytest.approx(
            sum(record["echo"] for record in records) / len(records)
        )

    # The whole payload must be JSON-serializable (mx scalars converted).
    json.dumps(probe)
    assert "accept@1" in module.format_summary_table(probe["variants"])


def test_probe_positions_cap_to_prompt_length(tmp_path: Path) -> None:
    module = _load_module()
    runtime, _rt_ar, rt_mtp = streamed_fixtures._injected_runtime_pair(tmp_path)
    try:
        probe = module.probe_draft_ranks(rt_mtp, PROMPT, probe_positions=200)
        with pytest.raises(ValueError, match="at least 3"):
            module.probe_draft_ranks(rt_mtp, [1, 2], probe_positions=1)
    finally:
        runtime.close()
    assert probe["probe_positions"] == len(PROMPT) - 2
    assert probe["probe_positions_capped"] is True
    assert probe["first_position"] == 0
    records = probe["variants"]["post_norm"]["records"]
    assert [record["position"] for record in records] == list(
        range(0, len(PROMPT) - 2)
    )


def test_chunked_prefill_matches_single_chunk(tmp_path: Path) -> None:
    module = _load_module()
    runtime, _rt_ar, rt_mtp = streamed_fixtures._injected_runtime_pair(tmp_path)
    try:
        whole = module.probe_draft_ranks(
            rt_mtp, PROMPT, probe_positions=6, prefill_chunk=256
        )
        chunked = module.probe_draft_ranks(
            rt_mtp, PROMPT, probe_positions=6, prefill_chunk=3, draft_batch=2
        )
    finally:
        runtime.close()
    for variant in ("post_norm", "pre_norm"):
        whole_records = whole["variants"][variant]["records"]
        chunked_records = chunked["variants"][variant]["records"]
        assert [r["position"] for r in whole_records] == [
            r["position"] for r in chunked_records
        ]
        assert [r["rank"] for r in whole_records] == [
            r["rank"] for r in chunked_records
        ]
        assert [r["argmax_token"] for r in whole_records] == [
            r["argmax_token"] for r in chunked_records
        ]


def test_echo_detection_flags_a_parroting_head(tmp_path: Path) -> None:
    module = _load_module()
    runtime, _rt_ar, rt_mtp = streamed_fixtures._injected_runtime_pair(tmp_path)
    try:
        vocab = 128
        original_forward = rt_mtp.model.mtp_forward

        def echoing_mtp_forward(hidden_states, next_token_ids, **kwargs):
            result = original_forward(hidden_states, next_token_ids, **kwargs)
            logits = result[0] if isinstance(result, tuple) else result
            # Argmax exactly at the head's own input token: pure echo.
            forced = (
                (mx.arange(vocab)[None, None, :] == next_token_ids[..., None])
                .astype(logits.dtype)
                * 10.0
            )
            if isinstance(result, tuple):
                return forced, result[1]
            return forced

        rt_mtp.model.mtp_forward = echoing_mtp_forward
        probe = module.probe_draft_ranks(rt_mtp, PROMPT, probe_positions=6)
    finally:
        runtime.close()
    for payload in probe["variants"].values():
        assert payload["summary"]["echo_rate"] == 1.0
        for record in payload["records"]:
            assert record["echo"] is True
            assert record["argmax_token"] == record["input_token"]
            # An echoing head can only be "right" when the prompt repeats
            # itself; this strictly increasing prompt never does.
            assert record["rank"] > 0
        assert payload["summary"]["acceptance_at_1"] == 0.0
