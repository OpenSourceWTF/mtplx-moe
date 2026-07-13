from __future__ import annotations

import importlib.util
import signal
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "run_issue30_starvation_attribution.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_issue30_starvation_attribution", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_configuration_summary(*, telemetry: bool = False) -> dict[str, object]:
    return {
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "requested_concurrency": 1,
        "execution_lane": "continuous-batch-ar",
        "performance_settings": {
            "runtime_config": {
                "model_key": "hy3-q4",
                "resource_telemetry": telemetry,
            },
            "prompt_identity": {
                "content_bytes": 10,
                "content_sha256": "a" * 64,
                "token_count": 3,
                "token_sha256": "b" * 64,
            },
            "generation": {"generation_profile": "deterministic", "max_tokens": 256},
            "model_artifact": {
                "method": "manifest_plus_executable_resident_content_v1",
                "expert_payload": {"sha256": "c" * 64, "size": 123},
                "manifest": {
                    "content_sha256": "d" * 64,
                    "declared_manifest_sha256": "e" * 64,
                    "model_key": "hy3-q4",
                    "source_repo": "pipenetwork/Hy3-4bit",
                    "source_revision": "f" * 40,
                },
                "harness_source": {
                    "dirty": False,
                    "git_head": "1" * 40,
                    "source_sha256": "2" * 64,
                },
            },
            "mtp": {"enabled": False, "precision": "bf16"},
            "prompt_options": {"chat": True, "enable_thinking": False},
            "sampler": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
            "scheduler": {
                "execution_lane": "continuous-batch-ar",
                "requested_concurrency": 1,
                "workload_shape": "static",
            },
            "seed": 0,
        },
    }


def _valid_cache_payload(*, read_ns: int = 10, read_bytes: int = 123):
    cache = {
        "bytes_read": read_bytes,
        "evictions": 1,
        "expert_hits": 10,
        "expert_misses": 2,
        "expert_requests": 12,
        "hit_rate": 10 / 12,
        "persistent_loads": 2,
        "route_calls": 2,
        "shared_expert_assignments": 0,
        "transient_loads": 0,
        "unique_expert_requests": 12,
    }
    io = {
        "record_requests": 8,
        "source_record_requests": 0,
        "sidecar_record_requests": 8,
        "read_operations": 8,
        "python_preadv_invocations": 8,
        "preadv_bytes_returned": read_bytes,
        "native_positional_calls": 0,
        "native_bytes_returned": 0,
        "requested_bytes": read_bytes,
        "read_bytes": read_bytes,
        "read_ns": read_ns,
        "read_mib_per_second": read_bytes / max(1, read_ns),
        "short_reads": 0,
        "integrity_errors": 0,
        "cancellations": 0,
        "deadline_errors": 0,
        "io_errors": 0,
    }
    return {
        "runs": [
            {
                "streaming_after": {
                    "cache": cache,
                    "cache_by_phase": {
                        "prefill": dict(cache),
                        "decode": dict(cache),
                    },
                    "incremental_misses": {"routes": 2, "parts": 4},
                    "slots": {
                        "io": io,
                        "metrics": {
                            "load_failures": 0,
                            "completion_fence_failures": 0,
                            "active_routes": 0,
                        },
                        "states": {"loading": 0, "failed": 0},
                        "pins": 0,
                    },
                }
            }
        ]
    }


def test_issue30_campaign_pins_balanced_physical_order_and_exact_flags(
    tmp_path: Path,
) -> None:
    module = _load_module()

    assert module.CAMPAIGN_ORDER == (
        "off-p01",
        "on-p01",
        "on-p02",
        "off-p02",
        "off-p03",
        "on-p03",
        "on-p04",
        "off-p04",
    )

    common = {
        "repo": tmp_path,
        "model": tmp_path / "model",
        "label": "moe-runtime-issue30-off-p01-20260713T120000Z-deadbee",
        "output_dir": tmp_path / "off-p01",
    }
    off = module.build_benchmark_command(**common, telemetry=False)
    on = module.build_benchmark_command(
        **{**common, "label": common["label"].replace("off", "on")},
        telemetry=True,
    )

    assert "--no-resource-telemetry" in off
    assert "--resource-telemetry" not in off
    assert "--resource-telemetry" in on
    assert "--no-resource-telemetry" not in on
    assert on[on.index("--resource-sample-interval") + 1] == "0.25"
    assert on[on.index("--resource-max-samples") + 1] == "4096"
    assert on[on.index("--ssd-ceiling-gib-s") + 1] == "12.47"
    assert "--no-powermetrics" in on
    for command in (off, on):
        assert command[command.index("--seed") + 1] == "0"
        assert "--no-enable-thinking" in command
        assert command[command.index("--repeats") + 1] == "1"
        assert command[command.index("--route-trace-json") + 1].endswith("/routes.json")
        assert command[command.index("--output-json") + 1].endswith("/result.json")


def test_normalized_config_fingerprint_removes_only_telemetry_toggle() -> None:
    module = _load_module()
    base = {
        "model_key": "hy3-q4",
        "resource_telemetry": False,
        "cache_scope": "global",
    }
    enabled = {**base, "resource_telemetry": True}

    assert module.normalized_config_fingerprint(base) == (
        module.normalized_config_fingerprint(enabled)
    )
    assert module.normalized_config_fingerprint(base) != (
        module.normalized_config_fingerprint({**base, "cache_scope": "layer"})
    )


def test_normalized_configuration_covers_prompt_model_and_generation() -> None:
    module = _load_module()
    disabled = _valid_configuration_summary(telemetry=False)
    enabled = _valid_configuration_summary(telemetry=True)
    changed_prompt = _valid_configuration_summary(telemetry=False)
    changed_prompt["performance_settings"]["prompt_identity"]["content_sha256"] = (
        "9" * 64
    )

    assert module.normalized_configuration_fingerprint(
        disabled
    ) == module.normalized_configuration_fingerprint(enabled)
    assert module.normalized_configuration_fingerprint(
        disabled
    ) != module.normalized_configuration_fingerprint(changed_prompt)


def test_normalized_configuration_rejects_missing_required_identity() -> None:
    module = _load_module()
    summary = _valid_configuration_summary()
    del summary["performance_settings"]["prompt_identity"]

    with pytest.raises(RuntimeError, match="prompt_identity"):
        module.normalized_configuration_fingerprint(summary)


def test_exclusive_window_restores_qwen_and_releases_lane_after_failure() -> None:
    module = _load_module()
    calls: list[object] = []
    state = module.QwenState(
        loaded=True,
        models=("mtplx-qwen36-27b-optimized-speed",),
    )

    def fail_workload() -> None:
        calls.append("workload")
        raise RuntimeError("injected benchmark failure")

    with pytest.raises(RuntimeError, match="injected benchmark failure"):
        module.run_exclusive_window(
            fail_workload,
            acquire_lane=lambda: calls.append("acquire"),
            release_lane=lambda: calls.append("release"),
            capture_qwen=lambda: calls.append("capture") or state,
            stop_qwen=lambda captured: calls.append(("stop", captured)),
            restore_qwen=lambda captured: calls.append(("restore", captured)),
        )

    assert calls == [
        "acquire",
        "capture",
        ("stop", state),
        "workload",
        ("restore", state),
        "release",
    ]


def test_exclusive_window_releases_lane_if_qwen_capture_is_ambiguous() -> None:
    module = _load_module()
    calls: list[str] = []

    def fail_capture():
        calls.append("capture")
        raise RuntimeError("ambiguous qwen state")

    with pytest.raises(RuntimeError, match="ambiguous qwen state"):
        module.run_exclusive_window(
            lambda: calls.append("workload"),
            acquire_lane=lambda: calls.append("acquire"),
            release_lane=lambda: calls.append("release"),
            capture_qwen=fail_capture,
            stop_qwen=lambda _state: calls.append("stop"),
            restore_qwen=lambda _state: calls.append("restore"),
        )

    assert calls == ["acquire", "capture", "release"]


def test_benchmark_process_terminates_the_uv_process_group_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[object] = []

    class Process:
        pid = 4321

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    def popen(command, **kwargs):
        calls.append(("popen", command, kwargs))
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )
    monkeypatch.setattr(module, "_process_group_exists", lambda _pid: False)

    with pytest.raises(KeyboardInterrupt):
        module._run_benchmark_process(["uv", "run", "python"], cwd=tmp_path)

    assert calls[0][2]["start_new_session"] is True
    assert ("killpg", 4321, signal.SIGTERM) in calls
    assert ("wait", 10.0) in calls


def test_benchmark_process_kills_descendant_group_after_uv_leader_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[object] = []

    class Process:
        pid = 4321

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return 0

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )
    group_states = iter((True, False))
    monkeypatch.setattr(
        module,
        "_process_group_exists",
        lambda _pid: next(group_states),
    )

    with pytest.raises(KeyboardInterrupt):
        module._run_benchmark_process(["uv", "run", "python"], cwd=tmp_path)

    assert ("killpg", 4321, signal.SIGTERM) in calls
    assert ("killpg", 4321, signal.SIGKILL) in calls


def test_benchmark_process_reaps_killed_leader_before_group_exit_poll(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[object] = []

    class Process:
        pid = 4321
        reaped = False

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            if timeout == 10.0:
                raise module.subprocess.TimeoutExpired("uv", timeout)
            self.reaped = True
            return -signal.SIGKILL

    process = Process()
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(module, "_process_group_exists", lambda _pid: True)
    monkeypatch.setattr(module.os, "killpg", lambda *_args: None)

    def wait_for_group(_pid, *, timeout_seconds):
        calls.append(("group-wait", timeout_seconds))
        assert process.reaped is True
        return True

    monkeypatch.setattr(module, "_wait_for_process_group_exit", wait_for_group)

    with pytest.raises(KeyboardInterrupt):
        module._run_benchmark_process(["uv", "run", "python"], cwd=tmp_path)

    assert calls.index(("wait", 5.0)) < calls.index(("group-wait", 5.0))


def test_benchmark_process_rejects_descendants_after_clean_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[object] = []

    class Process:
        pid = 4321

        @staticmethod
        def wait(timeout=None):
            assert timeout is None
            return 0

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    group_states = iter((True, False))
    monkeypatch.setattr(
        module,
        "_process_group_exists",
        lambda _pid: next(group_states),
    )
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )

    with pytest.raises(RuntimeError, match="descendants remained"):
        module._run_benchmark_process(["uv", "run", "python"], cwd=tmp_path)

    assert ("killpg", 4321, signal.SIGTERM) in calls


def test_exclusive_cleanup_blocks_termination_signals_until_lane_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    calls: list[object] = []

    def pthread_sigmask(how, mask):
        calls.append(("sigmask", how, frozenset(mask)))
        return frozenset()

    monkeypatch.setattr(module.signal, "pthread_sigmask", pthread_sigmask)
    state = module.QwenState(loaded=False, models=())
    module.run_exclusive_window(
        lambda: calls.append("workload"),
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture_qwen=lambda: calls.append("capture") or state,
        stop_qwen=lambda _state: calls.append("stop"),
        restore_qwen=lambda _state: calls.append("restore"),
    )

    restore_index = calls.index("restore")
    release_index = calls.index("release")
    block_index = max(
        index
        for index, value in enumerate(calls)
        if isinstance(value, tuple)
        and value[:2] == ("sigmask", module.signal.SIG_BLOCK)
    )
    unblock_index = max(
        index
        for index, value in enumerate(calls)
        if isinstance(value, tuple)
        and value[:2] == ("sigmask", module.signal.SIG_SETMASK)
    )
    assert block_index < restore_index < release_index < unblock_index


def test_run_record_persists_argv_and_start_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(module, "_assert_clean_sha", lambda *_args: "a" * 40)
    monkeypatch.setattr(module, "_command_observation", lambda _command: {})

    def run_process(command, *, cwd):
        assert events and events[0][0] == "start"
        events.append(("launch", (command, cwd)))
        return 1

    monkeypatch.setattr(module, "_run_benchmark_process", run_process)

    record = module._run_record(
        repo=tmp_path,
        model=tmp_path / "model",
        campaign_dir=tmp_path / "campaign",
        variant="off-p01",
        sha="a" * 40,
        on_start=lambda value: events.append(("start", dict(value))),
    )

    persisted = events[0][1]
    assert persisted["variant"] == "off-p01"
    assert persisted["exit_code"] is None
    assert persisted["argv"]
    assert record["exit_code"] == 1


def test_cache_signature_ignores_io_timing_but_not_deterministic_counters() -> None:
    module = _load_module()

    assert module._cache_signature(
        _valid_cache_payload(read_ns=10)
    ) == module._cache_signature(_valid_cache_payload(read_ns=20))
    assert module._cache_signature(
        _valid_cache_payload(read_ns=10)
    ) != module._cache_signature(_valid_cache_payload(read_ns=10, read_bytes=124))


def test_cache_signature_rejects_missing_deterministic_counter() -> None:
    module = _load_module()
    payload = _valid_cache_payload()
    del payload["runs"][0]["streaming_after"]["slots"]["io"][
        "python_preadv_invocations"
    ]

    with pytest.raises(RuntimeError, match="python_preadv_invocations"):
        module._cache_signature(payload)


def test_parity_requires_one_distinct_raw_fingerprint_per_arm() -> None:
    module = _load_module()

    def records() -> list[dict[str, object]]:
        return [
            {
                "variant": variant,
                "telemetry": variant.startswith("on-"),
                "exit_code": 0,
                "raw_configuration_fingerprint": (
                    "raw-on" if variant.startswith("on-") else "raw-off"
                ),
                "normalized_configuration_fingerprint": "normalized",
                "stream_signature": "stream",
                "cache_signature": "cache",
                "route_signature": "route",
                "manifest_sha256": "manifest",
                "final_health_ok": True,
            }
            for variant in module.CAMPAIGN_ORDER
        ]

    valid = module._parity_summary(records())
    assert valid["all_selected_parity"] is True

    drifted = records()
    drifted[-1]["raw_configuration_fingerprint"] = "raw-off-drift"
    assert module._parity_summary(drifted)["all_selected_parity"] is False

    prompt_drift = records()
    prompt_drift[-1]["normalized_configuration_fingerprint"] = "prompt-drift"
    assert module._parity_summary(prompt_drift)["all_selected_parity"] is False

    unhealthy = records()
    for record in unhealthy:
        record["final_health_ok"] = False
    assert module._parity_summary(unhealthy)["all_selected_parity"] is False


def test_route_and_stream_evidence_fail_closed_when_missing() -> None:
    module = _load_module()
    manifest = "a" * 64
    route = {
        "expert_ids": [2],
        "layer": 1,
        "phase": "decode",
        "token_count": 1,
        "trace_epoch": 0,
        "decode_step": 0,
    }
    assert module._route_evidence(
        {"entries": [route], "manifest_sha256": manifest}
    ) == (module._json_fingerprint([route]), manifest)

    with pytest.raises(RuntimeError, match="route entries"):
        module._route_evidence({"entries": [], "manifest_sha256": manifest})
    with pytest.raises(RuntimeError, match="manifest SHA"):
        module._route_evidence({"entries": [route], "manifest_sha256": None})
    with pytest.raises(RuntimeError, match="stream list"):
        module._stream_signature({"runs": [{"streams": []}]})


def test_route_evidence_rejects_malformed_nonempty_entry() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="expert_ids"):
        module._route_evidence(
            {
                "entries": [
                    {
                        "layer": 1,
                        "phase": "decode",
                        "token_count": 1,
                        "trace_epoch": 0,
                        "decode_step": 0,
                    }
                ],
                "manifest_sha256": "a" * 64,
            }
        )


def test_stream_signature_rejects_malformed_nonempty_row() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="token_ids"):
        module._stream_signature(
            {
                "runs": [
                    {
                        "streams": [
                            {
                                "seed": 0,
                                "prompt_tokens": 3,
                                "completion_tokens": 1,
                                "finish_reason": "length",
                                "text": "answer",
                            }
                        ]
                    }
                ]
            }
        )


def test_hardware_profile_drops_stable_machine_identifiers() -> None:
    module = _load_module()
    sanitized = module._sanitize_hardware_profile(
        {
            "SPHardwareDataType": [
                {
                    "machine_name": "MacBook Pro",
                    "machine_model": "Mac17,6",
                    "chip_type": "Apple M5 Max",
                    "physical_memory": "128 GB",
                    "serial_number": "SECRET-SERIAL",
                    "platform_UUID": "SECRET-UUID",
                    "provisioning_UDID": "SECRET-UDID",
                }
            ]
        }
    )

    assert sanitized == {
        "machine_name": "MacBook Pro",
        "machine_model": "Mac17,6",
        "chip_type": "Apple M5 Max",
        "physical_memory": "128 GB",
    }


def test_qwen_final_state_must_equal_captured_state() -> None:
    module = _load_module()
    unloaded = module.QwenState(loaded=False, models=())
    loaded = module.QwenState(
        loaded=True,
        models=("mtplx-qwen36-27b-optimized-speed",),
    )

    module._assert_qwen_state_restored(unloaded, unloaded)
    with pytest.raises(RuntimeError, match="differs from the captured state"):
        module._assert_qwen_state_restored(unloaded, loaded)
