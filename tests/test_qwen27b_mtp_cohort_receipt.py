from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import qwen27b_mtp_cohort_receipt as receipt
from scripts.qwen27b_mtp_cohort_receipt import evaluate_promotion, summarize_cell


ROOT = Path(__file__).resolve().parents[1]


def test_receipt_module_import_does_not_require_mlx(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        """
import importlib.abc
import sys

class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
            raise ModuleNotFoundError(f"blocked {fullname}")
        return None

sys.meta_path.insert(0, BlockMLX())
""".lstrip(),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(blocker), str(ROOT)))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.qwen27b_mtp_cohort_receipt import summarize_cell; print('ok')",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_summarize_cell_reports_aggregate_and_per_request_rates() -> None:
    cell = summarize_cell(
        [
            {
                "request_id": "a",
                "completion_tokens": 256,
                "elapsed_s": 10.0,
                "ttft_s": 0.2,
                "decode_tok_s": 26.0,
            },
            {
                "request_id": "b",
                "completion_tokens": 256,
                "elapsed_s": 10.0,
                "ttft_s": 0.3,
                "decode_tok_s": 25.0,
            },
        ]
    )

    assert cell["aggregate_output_tok_s"] == 51.2
    assert cell["per_request_decode_tok_s"] == [26.0, 25.0]
    assert cell["max_ttft_s"] == 0.3


@pytest.mark.parametrize(
    "missing",
    [
        "request_id",
        "completion_tokens",
        "elapsed_s",
        "ttft_s",
        "decode_tok_s",
    ],
)
def test_summarize_cell_rejects_missing_required_field(missing: str) -> None:
    row = {
        "request_id": "a",
        "completion_tokens": 1,
        "elapsed_s": 1.0,
        "ttft_s": 0.1,
        "decode_tok_s": 1.0,
    }
    del row[missing]

    with pytest.raises(ValueError, match=missing):
        summarize_cell([row])


def _promotion_receipts() -> tuple[dict[str, object], dict[str, object]]:
    control: dict[str, object] = {
        "status": "complete",
        "metrics": {
            "c1": {
                "aggregate_output_tok_s": 100.0,
                "repeat_values": [99.0, 100.0, 101.0],
            },
            "c2": {
                "aggregate_output_tok_s": 100.0,
                "repeat_values": [99.0, 100.0, 101.0],
            },
            "c2_4k": {
                "max_ttft_s": 4.0,
                "repeat_values": [3.9, 4.0, 4.1],
            },
            "production": {
                "max_ttft_s": 8.0,
                "repeat_values": [7.9, 8.0, 8.1],
            },
            "long_prefill": {
                "prefill_tok_s": 200.0,
                "repeat_values": [199.0, 200.0, 201.0],
            },
        },
    }
    candidate: dict[str, object] = {
        "status": "complete",
        "metrics": {
            "c1": {
                "aggregate_output_tok_s": 99.0,
                "repeat_values": [98.0, 99.0, 100.0],
            },
            "c2": {
                "aggregate_output_tok_s": 135.0,
                "repeat_values": [134.0, 135.0, 136.0],
            },
            "c2_4k": {
                "max_ttft_s": 4.2,
                "repeat_values": [4.1, 4.2, 4.3],
            },
            "production": {
                "max_ttft_s": 8.4,
                "repeat_values": [8.3, 8.4, 8.5],
            },
            "long_prefill": {
                "prefill_tok_s": 190.0,
                "repeat_values": [189.0, 190.0, 191.0],
                "short_admitted_between_chunks": True,
                "prefill_chunk_tokens": 1024,
            },
        },
        "validation": {
            "token_parity": True,
            "acceptance_parity": True,
            "cache_isolation": True,
            "session_isolation": True,
            "streaming": True,
            "constraint": True,
            "tool": True,
            "cancellation": True,
        },
        "scheduler_lanes": [
            "mtp_cohort_width_1",
            "mtp_cohort_width_2",
        ],
        "fallback_reasons": [],
        "retry_reasons": [],
    }
    return control, candidate


def test_evaluate_promotion_accepts_exact_thresholds() -> None:
    control, candidate = _promotion_receipts()

    result = evaluate_promotion(control, candidate)

    assert result["solo_ratio"] == pytest.approx(0.99)
    assert result["pair_ratio"] == pytest.approx(1.35)
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (
            lambda _control, candidate: candidate["metrics"]["c1"].__setitem__(
                "aggregate_output_tok_s", 98.9
            ),
            "solo_throughput",
        ),
        (
            lambda _control, candidate: candidate["metrics"]["c2"].__setitem__(
                "aggregate_output_tok_s", 134.9
            ),
            "pair_throughput",
        ),
        (
            lambda _control, candidate: candidate["validation"].__setitem__(
                "token_parity", False
            ),
            "token_parity",
        ),
        (
            lambda _control, candidate: candidate["validation"].__setitem__(
                "acceptance_parity", False
            ),
            "acceptance_parity",
        ),
        (
            lambda _control, candidate: candidate["metrics"]["c1"].__setitem__(
                "repeat_values", [99.0, 100.0]
            ),
            "paired_repeats",
        ),
        (
            lambda _control, candidate: candidate["metrics"]["c2_4k"].__setitem__(
                "max_ttft_s", 4.21
            ),
            "c2_4k_ttft",
        ),
        (
            lambda _control, candidate: candidate["metrics"][
                "production"
            ].__setitem__("max_ttft_s", 8.41),
            "production_ttft",
        ),
        (
            lambda _control, candidate: candidate["validation"].__setitem__(
                "cache_isolation", False
            ),
            "cache_isolation",
        ),
        (
            lambda _control, candidate: candidate.__setitem__(
                "fallback_reasons", ["batch_size_gt_1"]
            ),
            "fallback_free",
        ),
        (
            lambda _control, candidate: candidate.__setitem__(
                "retry_reasons", ["stream_retry"]
            ),
            "retry_free",
        ),
        (
            lambda _control, candidate: candidate.__setitem__(
                "scheduler_lanes", ["mtp_cohort_width_1", "ar_batch"]
            ),
            "scheduler_lanes",
        ),
        (
            lambda _control, candidate: candidate["metrics"][
                "long_prefill"
            ].__setitem__("short_admitted_between_chunks", False),
            "prefill_overlap",
        ),
        (
            lambda _control, candidate: candidate["metrics"][
                "long_prefill"
            ].__setitem__("prefill_tok_s", 189.9),
            "long_prefill_throughput",
        ),
        (
            lambda _control, candidate: candidate["metrics"][
                "long_prefill"
            ].__setitem__("prefill_chunk_tokens", 512),
            "prefill_chunk_tokens",
        ),
    ],
)
def test_evaluate_promotion_rejects_failed_gate(
    mutation: Any,
    failure: str,
) -> None:
    control, candidate = _promotion_receipts()
    mutation(control, candidate)

    result = evaluate_promotion(control, candidate)

    assert result["passed"] is False
    assert failure in result["failures"]


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_harness_provenance_records_stable_hash_and_dirty_marker(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    harness = repo / "receipt.py"
    harness.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "receipt@example.invalid")
    _git(repo, "config", "user.name", "Receipt Test")
    _git(repo, "add", "receipt.py")
    _git(repo, "commit", "-qm", "base")

    clean = receipt._harness_provenance(repo, harness)

    assert clean["harness_sha256"] == hashlib.sha256(harness.read_bytes()).hexdigest()
    assert clean["worktree_dirty"] is False
    assert clean["git_status_porcelain"] == ""

    harness.write_text("VALUE = 2\n", encoding="utf-8")
    dirty = receipt._harness_provenance(repo, harness)

    assert dirty["worktree_dirty"] is True
    assert dirty["harness_sha256"] != clean["harness_sha256"]
    assert dirty["working_tree_diff_sha256"] != clean["working_tree_diff_sha256"]
    assert dirty["provenance_identity_sha256"] != clean["provenance_identity_sha256"]


def test_control_receipt_must_match_candidate_harness_bytes(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "control-stale.json"
    matching = tmp_path / "control-matching.json"
    base = {
        "status": "complete",
        "mode": "control",
        "harness_provenance": {
            "worktree_dirty": False,
        },
    }
    stale.write_text(
        json.dumps(
            {
                **base,
                "harness_provenance": {
                    **base["harness_provenance"],
                    "harness_sha256": "stale",
                },
            }
        ),
        encoding="utf-8",
    )
    matching.write_text(
        json.dumps(
            {
                **base,
                "harness_provenance": {
                    **base["harness_provenance"],
                    "harness_sha256": "current",
                },
            }
        ),
        encoding="utf-8",
    )

    path, _payload = receipt._load_control_receipt(
        str(tmp_path / "control-*.json"),
        harness_sha256="current",
    )

    assert path == matching.resolve()


def test_session_reuse_messages_repeat_the_exact_committed_prompt() -> None:
    assert receipt._session_reuse_messages("prompt") == [
        {"role": "user", "content": "prompt"}
    ]


def test_session_probe_replays_one_short_constrained_request() -> None:
    headers = {"X-MTPLX-Session-ID": "session"}
    seed = receipt._session_probe_request_spec(
        request_id="seed",
        extra_headers=headers,
    )
    assistant_content = '{\n  "status": "ready"\n}'
    followup = receipt._session_probe_request_spec(
        request_id="followup",
        extra_headers=headers,
        assistant_content=assistant_content,
    )

    assert seed["max_tokens"] == 64
    assert seed["require_max_tokens"] is False
    assert seed["cache_mode"] is None
    assert seed["body_overrides"] == {
        "response_format": {"type": "json_object"},
    }
    assert seed["messages"] == receipt._session_reuse_messages(
        receipt.SESSION_PROBE_PROMPT
    )
    assert followup["messages"] == [
        *seed["messages"],
        {"role": "assistant", "content": assistant_content},
        {
            "role": "user",
            "content": "Return a JSON object with the single key status set to complete.",
        },
    ]
    assert followup["extra_headers"] == headers
    assert followup["body_overrides"] == seed["body_overrides"]


def test_wait_for_session_prefix_observes_async_postcommit_entry() -> None:
    health_rows = iter(
        [
            {
                "session_bank": {"prefixes": []},
                "scheduler": {"active_lane": "mtp_cohort_idle"},
            },
            {
                "session_bank": {
                    "prefixes": [
                        {
                            "session_id": "other",
                            "prefix_len": 40,
                        },
                        {
                            "session_id": "target",
                            "prefix_len": 34,
                        },
                    ]
                },
                "scheduler": {"active_lane": "mtp_cohort_idle"},
            },
        ]
    )
    clock_rows = iter([0.0, 0.1, 0.2])
    sleeps: list[float] = []

    observed = receipt._wait_for_session_prefix(
        base_url="http://127.0.0.1:18081",
        session_id="target",
        process=SimpleNamespace(poll=lambda: None),
        timeout_s=5.0,
        http_json=lambda _url, **_kwargs: next(health_rows),
        monotonic=lambda: next(clock_rows),
        sleep=sleeps.append,
    )

    assert observed == {
        "session_id": "target",
        "prefix_len": 34,
        "polls": 2,
        "elapsed_s": pytest.approx(0.2),
    }
    assert sleeps == [0.05]


def test_constraint_validation_ignores_reasoning_and_requires_completed_json() -> None:
    row = {
        "stream_tokens": [
            {"kind": "reasoning_content", "text": "not json"},
            {"kind": "content", "text": '{"status":"ok",'},
            {"kind": "content", "text": '"rows":2}'},
        ],
        "mtplx_stats": {
            "constraint_active": True,
            "constraint_completed": True,
        },
    }

    assert receipt._constraint_row_valid(row) is True

    row["mtplx_stats"]["constraint_completed"] = False
    assert receipt._constraint_row_valid(row) is False


def test_receipt_requests_opt_in_to_deterministic_client_controls() -> None:
    headers = receipt._stream_request_headers(
        cache_mode="bypass",
        extra_headers=None,
    )

    assert headers["X-MTPLX-Allow-Client-Controls"] == "1"


def test_memory_receipt_falls_back_to_footprint() -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        if command[0] == "/usr/bin/vmmap":
            return SimpleNamespace(returncode=1, stdout="", stderr="vmmap failed")
        return SimpleNamespace(returncode=0, stdout="footprint: 12 GB\n", stderr="")

    result = receipt._memory_receipt(123, runner=run)

    assert result["tool"] == "footprint"
    assert result["returncode"] == 0
    assert calls == [
        ["/usr/bin/vmmap", "-summary", "123"],
        ["/usr/bin/footprint", "-p", "123"],
    ]
    assert len(result["attempts"]) == 2


def test_memory_receipt_rejects_when_both_tools_fail() -> None:
    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"{command[0]} failed",
        )

    with pytest.raises(RuntimeError, match="memory receipt failed"):
        receipt._memory_receipt(123, runner=run)


def _valid_health() -> dict[str, object]:
    return {
        "ok": True,
        "model_path": str(receipt.MODEL_PATH),
        "generation_mode": "mtp",
        "depth": 2,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "profile": {"name": "turbo"},
        "scheduler": {
            "mode": "serial",
            "config": {"max_active_requests": 2},
        },
    }


def test_final_health_is_validated() -> None:
    unhealthy = _valid_health()
    unhealthy["ok"] = False

    with pytest.raises(RuntimeError, match="server health contract failed"):
        receipt._capture_final_health(
            "http://127.0.0.1:18081",
            18081,
            http_json=lambda _url: unhealthy,
        )


def test_candidate_server_command_freezes_cohort_and_prefill_geometry() -> None:
    command = receipt._server_command(
        ROOT,
        18081,
        scheduler_mode="mtp_cohort_experimental",
    )

    assert command[command.index("--scheduler-mode") + 1] == (
        "mtp_cohort_experimental"
    )
    assert "--experimental-mtp-cohorts" in command
    assert command[command.index("--max-active-requests") + 1] == "2"
    assert command[command.index("--decode-batch-max") + 1] == "2"
    assert command[command.index("--batch-wait-ms") + 1] == "0"
    assert command[command.index("--prefill-chunk-tokens") + 1] == "1024"


def test_candidate_health_requires_installed_path_b() -> None:
    health = _valid_health()
    health["scheduler"] = {
        "mode": "mtp_cohort_experimental",
        "config": {
            "max_active_requests": 2,
            "decode_batch_max": 2,
            "batch_wait_ms": 0.0,
            "prefill_chunk_tokens": 1024,
            "experimental_mtp_cohorts": True,
        },
        "path": "path_b",
        "path_b": {
            "installed": True,
            "experimental_mtp_cohorts": True,
        },
    }

    receipt._assert_health_contract(
        health,
        18081,
        scheduler_mode="mtp_cohort_experimental",
    )

    health["scheduler"]["config"]["prefill_chunk_tokens"] = 512
    with pytest.raises(RuntimeError, match="prefill_chunk_tokens"):
        receipt._assert_health_contract(
            health,
            18081,
            scheduler_mode="mtp_cohort_experimental",
        )


def test_candidate_rows_require_native_mtp_without_fallback_or_retry() -> None:
    rows = [
        {
            "request_id": "a",
            "mtplx_stats": {
                "generation_mode": "mtp",
                "scheduler_lane": "mtp_cohort",
                "mtp_disabled_reason": None,
                "tool_fed_empty_retry_attempted": False,
            },
        }
    ]

    assert receipt._candidate_row_failures(rows) == []

    rows[0]["mtplx_stats"]["scheduler_lane"] = "ar_batch"
    rows[0]["mtplx_stats"]["mtp_disabled_reason"] = "batch_size_gt_1"
    rows[0]["mtplx_stats"]["tool_fed_empty_retry_attempted"] = True

    failures = receipt._candidate_row_failures(rows)

    assert any("scheduler_lane=ar_batch" in item for item in failures)
    assert any("mtp_disabled_reason=batch_size_gt_1" in item for item in failures)
    assert any("tool_fed_empty_retry_attempted" in item for item in failures)


class _FakeProcess:
    def __init__(self, pid: int, poll_result: int | None) -> None:
        self.pid = pid
        self.returncode = poll_result
        self._poll_result = poll_result

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return int(self.returncode or 0)


def test_capture_failure_stops_spawned_group_frees_port_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(500, None)
    port = {"occupied": False}
    signals: list[tuple[int, int]] = []
    clock = {"value": 0.0}

    def popen(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        port["occupied"] = True
        return process

    def killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        process._poll_result = -sig
        process.returncode = -sig
        port["occupied"] = False

    def monotonic() -> float:
        clock["value"] += 100.0
        return clock["value"]

    monkeypatch.setattr(
        receipt,
        "_harness_provenance",
        lambda _worktree, _harness: {
            "worktree_head": "test-head",
            "worktree_dirty": False,
            "git_status_porcelain": "",
        },
    )
    monkeypatch.setattr(receipt, "_run_contract_subprocess", lambda **_kwargs: {})
    monkeypatch.setattr(receipt.subprocess, "Popen", popen)
    monkeypatch.setattr(
        receipt,
        "_capture_owned_process_group",
        lambda _process: (_ for _ in ()).throw(
            RuntimeError("detailed ownership capture failed")
        ),
    )
    monkeypatch.setattr(receipt.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(receipt.os, "getsid", lambda pid: pid)
    monkeypatch.setattr(receipt.os, "killpg", killpg)
    monkeypatch.setattr(receipt, "_read_group_members", lambda _pgid: [])
    monkeypatch.setattr(
        receipt,
        "_port_is_free",
        lambda _host, _port: not port["occupied"],
    )
    monkeypatch.setattr(receipt.time, "monotonic", monotonic)

    output = tmp_path / "capture-failure.json"
    exit_code = receipt._run_main(
        SimpleNamespace(
            worktree=str(ROOT),
            server_port=18081,
            mode="control",
            repeats=3,
            output=str(output),
            server_timeout_s=1.0,
        )
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert signals == [(500, signal.SIGTERM)]
    assert port["occupied"] is False
    assert saved["status"] == "failed"
    assert saved["dedicated_port_free_after"] is True
    assert "detailed ownership capture failed" in saved["error"]
    assert saved["server"]["emergency_shutdown"]["group_exited"] is True


def test_owned_group_stops_lingering_child_after_leader_exits() -> None:
    process = _FakeProcess(100, 0)
    owner = receipt.OwnedProcessGroup(
        leader_pid=100,
        pgid=100,
        sid=100,
        leader_start="leader-start",
    )
    child = receipt.ProcessIdentity(
        pid=101,
        pgid=100,
        sid=100,
        start="child-start",
    )
    groups = iter(([child], []))
    signals: list[tuple[int, int]] = []

    result = receipt._stop_owned_process_group(
        process,
        owner,
        identity_reader=lambda _pid: None,
        group_members=lambda _pgid: next(groups),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
        term_timeout_s=0.0,
        kill_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert signals == [(100, signal.SIGTERM)]
    assert result["group_exited"] is True
    assert result["leader_exited_before_shutdown"] is True


def test_owned_group_escalates_when_sigterm_does_not_exit() -> None:
    process = _FakeProcess(200, None)
    owner = receipt.OwnedProcessGroup(
        leader_pid=200,
        pgid=200,
        sid=200,
        leader_start="leader-start",
    )
    leader = receipt.ProcessIdentity(
        pid=200,
        pgid=200,
        sid=200,
        start="leader-start",
    )
    groups = iter(([leader], [leader], []))
    signals: list[tuple[int, int]] = []

    result = receipt._stop_owned_process_group(
        process,
        owner,
        identity_reader=lambda _pid: leader,
        group_members=lambda _pgid: next(groups),
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
        term_timeout_s=0.0,
        kill_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert signals == [
        (200, signal.SIGTERM),
        (200, signal.SIGKILL),
    ]
    assert result["group_exited"] is True
    assert result["escalated"] is True


def test_owned_group_never_signals_reused_identity() -> None:
    process = _FakeProcess(300, None)
    owner = receipt.OwnedProcessGroup(
        leader_pid=300,
        pgid=300,
        sid=300,
        leader_start="original-start",
    )
    reused = receipt.ProcessIdentity(
        pid=300,
        pgid=300,
        sid=300,
        start="reused-start",
    )
    signals: list[tuple[int, int]] = []

    with pytest.raises(RuntimeError, match="identity changed"):
        receipt._stop_owned_process_group(
            process,
            owner,
            identity_reader=lambda _pid: reused,
            group_members=lambda _pgid: [reused],
            killpg=lambda pgid, sig: signals.append((pgid, sig)),
            term_timeout_s=0.0,
            kill_timeout_s=0.0,
            sleep=lambda _seconds: None,
        )

    assert signals == []


def test_owned_group_uses_exact_group_snapshot_when_leader_scan_misses() -> None:
    process = _FakeProcess(350, None)
    owner = receipt.OwnedProcessGroup(
        leader_pid=350,
        pgid=350,
        sid=350,
        leader_start="leader-start",
    )
    leader = receipt.ProcessIdentity(
        pid=350,
        pgid=350,
        sid=350,
        start="leader-start",
    )
    groups = iter(([leader], []))
    signals: list[tuple[int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        process._poll_result = -sig
        process.returncode = -sig

    result = receipt._stop_owned_process_group(
        process,
        owner,
        identity_reader=lambda _pid: None,
        group_members=lambda _pgid: next(groups),
        killpg=killpg,
        term_timeout_s=0.0,
        kill_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert signals == [(350, signal.SIGTERM)]
    assert result["group_exited"] is True


def test_owned_group_treats_empty_group_as_exited_when_popen_is_stale() -> None:
    process = _FakeProcess(375, None)
    owner = receipt.OwnedProcessGroup(
        leader_pid=375,
        pgid=375,
        sid=375,
        leader_start="leader-start",
    )
    signals: list[tuple[int, int]] = []

    result = receipt._stop_owned_process_group(
        process,
        owner,
        identity_reader=lambda _pid: None,
        group_members=lambda _pgid: [],
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
        term_timeout_s=0.0,
        kill_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert signals == []
    assert result["group_exited"] is True


def test_finalize_receipt_returns_nonzero_when_port_remains_occupied(
    tmp_path: Path,
) -> None:
    output = tmp_path / "receipt.json"
    payload: dict[str, object] = {"status": "complete", "server": {}}

    exit_code = receipt._finalize_receipt(
        payload,
        output=output,
        process=None,
        owned_group=None,
        host="127.0.0.1",
        port=18081,
        port_is_free=lambda _host, _port: False,
        port_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert saved["status"] == "failed"
    assert saved["dedicated_port_free_after"] is False


def test_finalize_receipt_returns_nonzero_when_shutdown_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "receipt.json"
    payload: dict[str, object] = {"status": "complete", "server": {}}
    process = _FakeProcess(400, None)
    owner = receipt.OwnedProcessGroup(
        leader_pid=400,
        pgid=400,
        sid=400,
        leader_start="leader-start",
    )

    exit_code = receipt._finalize_receipt(
        payload,
        output=output,
        process=process,
        owned_group=owner,
        host="127.0.0.1",
        port=18081,
        stop_group=lambda _process, _owner: (_ for _ in ()).throw(
            RuntimeError("shutdown failed")
        ),
        port_is_free=lambda _host, _port: True,
        port_timeout_s=0.0,
        sleep=lambda _seconds: None,
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert saved["status"] == "failed"
    assert "shutdown failed" in saved["shutdown_error"]
