from __future__ import annotations

import importlib.util
import http.server
import json
import os
import plistlib
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import mtplx.qwen_guard as qwen_guard_module
from mtplx.qwen_guard import (
    EXPECTED_QWEN_MODELS,
    CommandResult,
    QwenState,
    qwen_stopped_for_mlx,
)


UID = os.getuid()
SERVICE = f"gui/{UID}/com.tea.qwen"
DOMAIN = f"gui/{UID}"
PROCESS_PATTERN = "mtplx.server.openai.*Qwen3.6"


@pytest.fixture(autouse=True)
def _isolated_guard_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "isolated-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _write_plist(root: Path) -> Path:
    launcher = root / "start-qwen-test.sh"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    path = root / "com.tea.qwen.plist"
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": [str(launcher)],
            }
        )
    )
    path.chmod(0o644)
    return path


class FakeQwen:
    def __init__(
        self,
        *,
        loaded: bool,
        models: tuple[str, ...] | None,
        processes: tuple[int, ...],
    ) -> None:
        self.loaded = loaded
        self.models = models
        self.processes = processes
        self.commands: list[tuple[str, ...]] = []
        self.clock = 0.0
        self.bootstrap_ready = True
        self.bootstrap_models = EXPECTED_QWEN_MODELS
        self.bootstrap_processes = (909,)
        self.launch_inputs: list[tuple[str, Path, dict, bytes | None]] = []
        self.command_timeouts: list[float | None] = []
        self.api_timeouts: list[float] = []

    def run_command(
        self,
        command: tuple[str, ...],
        timeout: float | None = None,
    ) -> CommandResult:
        command = tuple(command)
        self.commands.append(command)
        self.command_timeouts.append(timeout)
        if command == ("launchctl", "print", SERVICE):
            return CommandResult(0 if self.loaded else 113, "", "")
        if command == ("pgrep", "-f", PROCESS_PATTERN):
            if self.processes:
                return CommandResult(
                    0,
                    "\n".join(str(pid) for pid in self.processes) + "\n",
                    "",
                )
            return CommandResult(1, "", "")
        if command[:3] == ("launchctl", "bootout", DOMAIN):
            self._record_launch_input(command)
            self.loaded = False
            self.models = None
            self.processes = ()
            return CommandResult(0, "", "")
        if command[:3] == ("launchctl", "bootstrap", DOMAIN):
            self._record_launch_input(command)
            self.loaded = True
            if self.bootstrap_ready:
                self.models = self.bootstrap_models
                self.processes = self.bootstrap_processes
            else:
                self.models = None
                self.processes = ()
            return CommandResult(0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    def _record_launch_input(self, command: tuple[str, ...]) -> None:
        path = Path(command[3])
        payload = plistlib.loads(path.read_bytes())
        arguments = payload["ProgramArguments"]
        wrapper_payload = None
        if len(arguments) == 1:
            wrapper_payload = Path(arguments[0]).read_bytes()
        self.launch_inputs.append((command[1], path, payload, wrapper_payload))

    def fetch_models(self, url: str, timeout: float) -> tuple[str, ...] | None:
        assert url == "http://qwen.test/v1/models"
        assert timeout > 0
        self.api_timeouts.append(timeout)
        return self.models

    def monotonic(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.clock += seconds


def _guard(plist: Path, fake: FakeQwen, *, timeout: float = 2.0):
    return qwen_stopped_for_mlx(
        plist=plist,
        api_url="http://qwen.test/v1/models",
        timeout_seconds=timeout,
        _run_command=fake.run_command,
        _fetch_models=fake.fetch_models,
        _monotonic=fake.monotonic,
        _sleep=fake.sleep,
        _getuid=lambda: UID,
    )


def test_loaded_qwen_is_stopped_and_exactly_restored(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with _guard(plist, fake) as initial:
        assert initial == QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
        assert fake.loaded is False
        assert fake.models is None
        assert fake.processes == ()

    assert fake.loaded is True
    assert fake.models == EXPECTED_QWEN_MODELS
    assert fake.processes == (909,)
    bootout, bootstrap = fake.launch_inputs
    assert bootout[0] == "bootout"
    assert bootstrap[0] == "bootstrap"
    assert bootout[1] == bootstrap[1]
    assert bootout[1] != plist


def test_initially_unloaded_qwen_remains_unloaded(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(loaded=False, models=None, processes=())

    with _guard(plist, fake) as initial:
        assert initial == QwenState(loaded=False, models=())

    assert not any(command[1] in {"bootout", "bootstrap"} for command in fake.commands)
    assert fake.loaded is False
    assert fake.models is None
    assert fake.processes == ()


@pytest.mark.parametrize(
    ("loaded", "models", "processes"),
    (
        (True, None, (101,)),
        (True, EXPECTED_QWEN_MODELS, ()),
        (True, ("unexpected-model",), (101,)),
        (False, EXPECTED_QWEN_MODELS, ()),
        (False, None, (101,)),
        (False, (), ()),
    ),
)
def test_ambiguous_service_api_process_state_fails_closed(
    tmp_path: Path,
    loaded: bool,
    models: tuple[str, ...] | None,
    processes: tuple[int, ...],
) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(loaded=loaded, models=models, processes=processes)

    with pytest.raises(RuntimeError, match="ambiguous|exact"):
        with _guard(plist, fake):
            raise AssertionError("ambiguous state must not yield")

    assert not any(command[1] in {"bootout", "bootstrap"} for command in fake.commands)


@pytest.mark.parametrize(
    "error",
    (RuntimeError("child failed"), KeyboardInterrupt("child interrupted")),
)
def test_body_exception_or_keyboard_interrupt_restores_loaded_qwen(
    tmp_path: Path,
    error: BaseException,
) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(type(error), match=str(error)):
        with _guard(plist, fake):
            raise error

    assert fake.loaded is True
    assert fake.models == EXPECTED_QWEN_MODELS
    assert fake.processes


def test_restore_timeout_overrides_body_result(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )
    fake.bootstrap_ready = False

    with pytest.raises(RuntimeError, match="restore.*exact model|timed out"):
        with _guard(plist, fake, timeout=0.5):
            pass

    assert any(
        command[:3] == ("launchctl", "bootstrap", DOMAIN) for command in fake.commands
    )


def test_wrong_model_list_never_counts_as_restored(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )
    fake.bootstrap_models = ("mtplx-qwen36-27b-other",)

    with pytest.raises(RuntimeError, match="restore.*exact model|timed out"):
        with _guard(plist, fake, timeout=0.5):
            pass


def test_initially_unloaded_service_appearing_during_window_is_booted_out(
    tmp_path: Path,
) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(loaded=False, models=None, processes=())

    with _guard(plist, fake):
        fake.loaded = True
        fake.models = EXPECTED_QWEN_MODELS
        fake.processes = (404,)

    assert fake.loaded is False
    assert fake.models is None
    assert fake.processes == ()
    assert any(
        command[:3] == ("launchctl", "bootout", DOMAIN) for command in fake.commands
    )


def test_timeout_is_propagated_to_every_state_observation(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(loaded=False, models=None, processes=())

    with _guard(plist, fake, timeout=0.01):
        pass

    assert fake.command_timeouts
    assert all(
        timeout is not None and 0 < timeout <= 0.01 for timeout in fake.command_timeouts
    )
    assert fake.api_timeouts
    assert all(0 < timeout <= 0.01 for timeout in fake.api_timeouts)


def test_default_command_runner_kills_stuck_observation_at_deadline() -> None:
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="timed out"):
        qwen_guard_module._run_command(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            timeout=0.01,
        )

    assert time.monotonic() - started < 0.5


def test_default_model_fetcher_enforces_total_slow_drip_deadline() -> None:
    payload = json.dumps(
        {"data": [{"id": EXPECTED_QWEN_MODELS[0]}]},
        separators=(",", ":"),
    ).encode()

    class SlowDripHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for value in payload:
                    self.wfile.write(bytes((value,)))
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowDripHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = qwen_guard_module._fetch_models(
            f"http://127.0.0.1:{server.server_port}/v1/models",
            timeout=0.05,
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert result is None
    assert elapsed < 0.15


def test_default_model_fetcher_returns_fast_valid_model_list() -> None:
    payload = json.dumps(
        {"data": [{"id": EXPECTED_QWEN_MODELS[0]}]},
        separators=(",", ":"),
    ).encode()

    class ImmediateHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ImmediateHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = qwen_guard_module._fetch_models(
            f"http://127.0.0.1:{server.server_port}/v1/models",
            timeout=1.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert result == EXPECTED_QWEN_MODELS


def test_plist_symlink_is_rejected_before_state_commands(tmp_path: Path) -> None:
    real = _write_plist(tmp_path)
    link_root = tmp_path / "link"
    link_root.mkdir()
    link = link_root / "com.tea.qwen.plist"
    link.symlink_to(real)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="symlink|regular|no-follow"):
        with _guard(link, fake):
            pass

    assert fake.commands == []


def test_plist_label_and_program_are_validated_before_state_commands(
    tmp_path: Path,
) -> None:
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.example.not-qwen",
                "ProgramArguments": ["/bin/echo", "not qwen"],
            }
        )
    )
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="Label|ProgramArguments|Qwen"):
        with _guard(plist, fake):
            pass

    assert fake.commands == []


def test_plist_program_key_is_rejected_before_state_mutation(tmp_path: Path) -> None:
    program = tmp_path / "qwen-program"
    program.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    program.chmod(0o755)
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "Program": str(program),
                "ProgramArguments": [
                    sys.executable,
                    "-m",
                    "mtplx.server.openai",
                    "--model",
                    "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                ],
            }
        )
    )
    plist.chmod(0o644)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="Program"):
        with _guard(plist, fake):
            replacement = tmp_path / "replacement-program"
            replacement.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            replacement.chmod(0o755)
            os.replace(replacement, program)

    assert fake.commands == []


def test_user_owned_direct_server_executable_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "qwen-python-shim"
    executable.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": [
                    str(executable),
                    "-m",
                    "mtplx.server.openai",
                    "--model",
                    "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                ],
            }
        )
    )
    plist.chmod(0o644)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="direct.*executable|system Python"):
        with _guard(plist, fake):
            replacement = tmp_path / "replacement-direct-executable"
            replacement.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            replacement.chmod(0o755)
            os.replace(replacement, executable)

    assert fake.commands == []


def test_unbound_system_python_direct_server_form_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": [
                    "/usr/bin/python3",
                    "-m",
                    "mtplx.server.openai",
                    "--model",
                    "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                ],
            }
        )
    )
    plist.chmod(0o644)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="direct.*executable|unsupported"):
        with _guard(plist, fake):
            raise AssertionError("unbound direct-server form must not yield")

    assert fake.commands == []


def test_owned_qwen_wrapper_plist_is_safely_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    launcher = tmp_path / "start-qwen-mtp.sh"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
                "ProgramArguments": [str(launcher)],
            }
        )
    )
    plist.chmod(0o644)
    fake = FakeQwen(loaded=False, models=None, processes=())

    with _guard(plist, fake) as state:
        assert state == QwenState(loaded=False, models=())


def test_plist_replacement_cannot_change_bootstrap_content(tmp_path: Path) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with _guard(plist, fake):
        replacement = tmp_path / "replacement.plist"
        replacement.write_bytes(
            plistlib.dumps(
                {
                    "Label": "com.tea.qwen",
                    "ProgramArguments": [
                        sys.executable,
                        "-m",
                        "mtplx.server.openai",
                        "--model",
                        "/models/Qwen3.6-SUBSTITUTED",
                    ],
                }
            )
        )
        replacement.chmod(0o644)
        os.replace(replacement, plist)

    bootout, bootstrap = fake.launch_inputs
    assert bootout[0] == "bootout"
    assert bootstrap[0] == "bootstrap"
    assert bootout[1] == bootstrap[1]
    assert bootout[1] != plist
    assert bootout[2] == bootstrap[2]
    assert "SUBSTITUTED" not in repr(bootstrap[2])
    assert not bootout[1].exists()


def test_wrapper_replacement_cannot_change_bootstrap_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    wrapper = tmp_path / "start-qwen-mtp.sh"
    original_payload = b"#!/bin/sh\n# ORIGINAL-QWEN-WRAPPER\nexit 0\n"
    wrapper.write_bytes(original_payload)
    wrapper.chmod(0o755)
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps({"Label": "com.tea.qwen", "ProgramArguments": [str(wrapper)]})
    )
    plist.chmod(0o644)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with _guard(plist, fake):
        replacement = tmp_path / "replacement-wrapper"
        replacement.write_bytes(b"#!/bin/sh\n# SUBSTITUTED-WRAPPER\nexit 0\n")
        replacement.chmod(0o755)
        os.replace(replacement, wrapper)

    bootout, bootstrap = fake.launch_inputs
    assert bootout[1] == bootstrap[1]
    assert bootout[3] == original_payload
    assert bootstrap[3] == original_payload
    snapshot_wrapper = Path(bootstrap[2]["ProgramArguments"][0])
    assert snapshot_wrapper != wrapper
    assert snapshot_wrapper.exists()
    assert snapshot_wrapper.read_bytes() == original_payload
    assert subprocess.run((str(snapshot_wrapper),), check=False).returncode == 0
    assert not bootstrap[1].exists()


def test_cached_wrapper_substitution_before_bootstrap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    wrapper = tmp_path / "start-qwen-mtp.sh"
    wrapper.write_bytes(b"#!/bin/sh\n# ORIGINAL-QWEN-WRAPPER\nexit 0\n")
    wrapper.chmod(0o755)
    plist = tmp_path / "com.tea.qwen.plist"
    plist.write_bytes(
        plistlib.dumps({"Label": "com.tea.qwen", "ProgramArguments": [str(wrapper)]})
    )
    plist.chmod(0o644)
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(RuntimeError, match="snapshot changed"):
        with _guard(plist, fake):
            cached_wrapper = Path(fake.launch_inputs[0][2]["ProgramArguments"][0])
            replacement = cached_wrapper.with_name("replacement-wrapper")
            replacement.write_bytes(b"#!/bin/sh\n# SUBSTITUTED-WRAPPER\nexit 99\n")
            replacement.chmod(0o500)
            os.replace(replacement, cached_wrapper)

    assert not any(item[0] == "bootstrap" for item in fake.launch_inputs)


def test_plist_with_symlink_ancestor_is_rejected_before_commands(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_plist(first)
    _write_plist(second)
    link = tmp_path / "current"
    link.symlink_to(first, target_is_directory=True)
    plist = link / "com.tea.qwen.plist"
    fake = FakeQwen(
        loaded=True,
        models=EXPECTED_QWEN_MODELS,
        processes=(101,),
    )

    with pytest.raises(ValueError, match="ancestor|symlink|no-follow"):
        with _guard(plist, fake):
            link.unlink()
            link.symlink_to(second, target_is_directory=True)

    assert fake.commands == []


def _load_cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts/run_with_qwen_stopped.py"
    spec = importlib.util.spec_from_file_location("run_with_qwen_stopped", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(
        self,
        events: list[object],
        *,
        returncode: int = 0,
        wait_error: BaseException | None = None,
        raise_signal: int | None = None,
    ) -> None:
        self.events = events
        self.returncode = returncode
        self.wait_error = wait_error
        self.raise_signal = raise_signal
        self.sent_signals: list[int] = []

    def wait(self) -> int:
        self.events.append("child-wait")
        if self.raise_signal is not None:
            signal.raise_signal(self.raise_signal)
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode

    def poll(self) -> int | None:
        return None if not self.sent_signals else -self.sent_signals[-1]

    def send_signal(self, signum: int) -> None:
        self.events.append(("child-signal", signum))
        self.sent_signals.append(signum)
        self.returncode = -signum


def _fake_guard(events: list[object], *, restore_error: Exception | None = None):
    @contextmanager
    def guard(**kwargs):
        events.append(("guard-enter", kwargs))
        try:
            yield QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
        finally:
            events.append("guard-restored")
            if restore_error is not None:
                raise restore_error

    return guard


def test_cli_requires_child_command_after_explicit_delimiter(tmp_path: Path) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)

    with pytest.raises(SystemExit) as error:
        module.parse_cli_args(["--plist", str(plist), "python", "job.py"])

    assert error.value.code == 2


@pytest.mark.parametrize("child_exit", (0, 7))
def test_cli_returns_child_exit_only_after_restoration(
    tmp_path: Path,
    child_exit: int,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []
    process = FakeProcess(events, returncode=child_exit)

    def popen(command, **kwargs):
        assert kwargs == {"start_new_session": True}
        events.append(("popen", command))
        return process

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py", "--flag"],
        _guard_factory=_fake_guard(events),
        _popen=popen,
    )

    assert result == child_exit
    assert events == [
        (
            "guard-enter",
            {
                "plist": plist,
                "api_url": "http://127.0.0.1:8080/v1/models",
                "timeout_seconds": 180.0,
            },
        ),
        ("popen", ("python", "job.py", "--flag")),
        "child-wait",
        "guard-restored",
    ]


def test_cli_restoration_failure_overrides_nonzero_child_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []
    process = FakeProcess(events, returncode=7)

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py"],
        _guard_factory=_fake_guard(
            events,
            restore_error=RuntimeError("restore timed out"),
        ),
        _popen=lambda command, **kwargs: process,
    )

    assert result == 1
    assert "restore timed out" in capsys.readouterr().err
    assert events[-1] == "guard-restored"


@pytest.mark.parametrize(
    ("wait_error", "expected_exit"),
    ((OSError("child launch failed"), 1), (KeyboardInterrupt(), 130)),
)
def test_cli_child_exception_or_keyboard_interrupt_restores_qwen(
    tmp_path: Path,
    wait_error: BaseException,
    expected_exit: int,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []
    process = FakeProcess(events, wait_error=wait_error)

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py"],
        _guard_factory=_fake_guard(events),
        _popen=lambda command, **kwargs: process,
    )

    assert result == expected_exit
    assert events[-1] == "guard-restored"


def test_cli_spawn_exception_restores_qwen(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []

    def failed_spawn(command, **kwargs):
        assert command == ("python", "job.py")
        assert kwargs == {"start_new_session": True}
        raise OSError("spawn failed")

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py"],
        _guard_factory=_fake_guard(events),
        _popen=failed_spawn,
    )

    assert result == 1
    assert "spawn failed" in capsys.readouterr().err
    assert events[-1] == "guard-restored"


def test_cli_real_sigterm_is_forwarded_then_restores_before_signal_exit(
    tmp_path: Path,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []
    process = FakeProcess(events, raise_signal=signal.SIGTERM)
    previous = signal.getsignal(signal.SIGTERM)

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py"],
        _guard_factory=_fake_guard(events),
        _popen=lambda command, **kwargs: process,
    )

    assert result == 128 + signal.SIGTERM
    assert ("child-signal", signal.SIGTERM) in events
    assert events[-1] == "guard-restored"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_cli_signal_during_child_launch_is_forwarded_after_spawn(
    tmp_path: Path,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    events: list[object] = []
    process = FakeProcess(events)

    def signal_during_popen(command, **kwargs):
        assert kwargs == {"start_new_session": True}
        signal.raise_signal(signal.SIGTERM)
        events.append(("popen", command))
        return process

    result = module.main(
        ["--plist", str(plist), "--", "python", "job.py"],
        _guard_factory=_fake_guard(events),
        _popen=signal_during_popen,
    )

    assert result == 128 + signal.SIGTERM
    assert ("child-signal", signal.SIGTERM) in events
    assert events[-1] == "guard-restored"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return int(path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise AssertionError(f"child did not write PID file: {path}")


def _cleanup_pid(path: Path) -> None:
    if not path.exists() or not path.stat().st_size:
        return
    pid = int(path.read_text(encoding="utf-8"))
    if _pid_is_alive(pid):
        os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while _pid_is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)


def _descendant_program(pid_file: Path) -> str:
    return (
        "import os,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "signal.signal(signal.SIGHUP,signal.SIG_IGN);"
        f"open({str(pid_file)!r},'w').write(str(os.getpid()));"
        "time.sleep(60)"
    )


def _guard_observing_descendant(
    events: list[object],
    pid_file: Path,
):
    @contextmanager
    def guard(**kwargs):
        events.append(("guard-enter", kwargs))
        try:
            yield QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
        finally:
            pid = _read_pid(pid_file)
            events.append(("guard-restored-with-descendant-alive", _pid_is_alive(pid)))

    return guard


@pytest.mark.parametrize("leader_exit", (0, 7))
def test_cli_reaps_surviving_descendant_before_restore_after_leader_exit(
    tmp_path: Path,
    leader_exit: int,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    pid_file = tmp_path / "descendant.pid"
    events: list[object] = []
    command = (
        "zsh",
        "-lc",
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(_descendant_program(pid_file))} & "
        f"while [[ ! -s {shlex.quote(str(pid_file))} ]]; do sleep 0.01; done; "
        f"exit {leader_exit}",
    )

    try:
        result = module.main(
            ["--plist", str(plist), "--", *command],
            _guard_factory=_guard_observing_descendant(events, pid_file),
            _popen=subprocess.Popen,
        )

        assert result == leader_exit
        assert events[-1] == ("guard-restored-with-descendant-alive", False)
        assert not _pid_is_alive(_read_pid(pid_file))
    finally:
        _cleanup_pid(pid_file)


@pytest.mark.parametrize(
    ("signum", "signal_name", "expected_exit"),
    (
        (signal.SIGTERM, "TERM", 143),
        (signal.SIGHUP, "HUP", 129),
    ),
)
def test_cli_signal_kills_ignoring_descendant_group_before_restore(
    tmp_path: Path,
    signum: int,
    signal_name: str,
    expected_exit: int,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    pid_file = tmp_path / "descendant.pid"
    events: list[object] = []
    command = (
        "zsh",
        "-lc",
        f"trap 'exit {expected_exit}' {signal_name}; "
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(_descendant_program(pid_file))} & "
        f"while [[ ! -s {shlex.quote(str(pid_file))} ]]; do sleep 0.01; done; "
        f"kill -{signal_name} {os.getpid()}; "
        "while true; do sleep 1; done",
    )

    try:
        result = module.main(
            ["--plist", str(plist), "--", *command],
            _guard_factory=_guard_observing_descendant(events, pid_file),
            _popen=subprocess.Popen,
        )

        assert result == 128 + signum
        assert result == expected_exit
        assert events[-1] == ("guard-restored-with-descendant-alive", False)
        assert not _pid_is_alive(_read_pid(pid_file))
    finally:
        _cleanup_pid(pid_file)


def test_cli_wait_exception_reaps_leader_before_eperm_group_probe_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_module()
    plist = _write_plist(tmp_path)
    pid_file = tmp_path / "descendant.pid"
    events: list[object] = []
    real_killpg = os.killpg
    holder: dict[str, object] = {}
    killed = False

    class WaitFailureProcess:
        def __init__(self, process: subprocess.Popen) -> None:
            self.process = process
            self.pid = process.pid
            self.reaped = False

        def wait(self) -> int:
            raise OSError("injected wait failure")

        def poll(self) -> int | None:
            result = self.process.poll()
            if result is not None:
                self.reaped = True
            return result

    def guarded_killpg(process_group_id: int, signum: int) -> None:
        nonlocal killed
        process = holder.get("process")
        if signum == signal.SIGKILL:
            killed = True
        if (
            signum == 0
            and killed
            and isinstance(process, WaitFailureProcess)
            and not process.reaped
        ):
            raise PermissionError("injected Darwin EPERM before leader reap")
        real_killpg(process_group_id, signum)

    command = (
        "zsh",
        "-lc",
        "trap '' TERM HUP; "
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(_descendant_program(pid_file))} & "
        f"while [[ ! -s {shlex.quote(str(pid_file))} ]]; do sleep 0.01; done; "
        "while true; do sleep 1; done",
    )

    def popen(command_value, **kwargs):
        process = subprocess.Popen(command_value, **kwargs)
        wrapped = WaitFailureProcess(process)
        holder["process"] = wrapped
        _read_pid(pid_file)
        return wrapped

    @contextmanager
    def guard(**_kwargs):
        try:
            yield QwenState(loaded=True, models=EXPECTED_QWEN_MODELS)
        finally:
            process = holder["process"]
            assert isinstance(process, WaitFailureProcess)
            try:
                real_killpg(process.pid, 0)
            except ProcessLookupError:
                group_exists = False
            except PermissionError:
                group_exists = True
            else:
                group_exists = True
            events.append(("guard-restored", process.reaped, group_exists))

    monkeypatch.setattr(module.os, "killpg", guarded_killpg)
    try:
        result = module.main(
            ["--plist", str(plist), "--", *command],
            _guard_factory=guard,
            _popen=popen,
        )

        assert result == 1
        assert events == [("guard-restored", True, False)]
    finally:
        process = holder.get("process")
        if isinstance(process, WaitFailureProcess):
            try:
                process.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.process.kill()
                process.process.wait(timeout=5.0)
            try:
                real_killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        _cleanup_pid(pid_file)
