from __future__ import annotations

import importlib.util
import os
import plistlib
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

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


def _write_plist(root: Path) -> Path:
    path = root / "com.tea.qwen.plist"
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": "com.tea.qwen",
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

    def run_command(self, command: tuple[str, ...]) -> CommandResult:
        command = tuple(command)
        self.commands.append(command)
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
            self.loaded = False
            self.models = None
            self.processes = ()
            return CommandResult(0, "", "")
        if command[:3] == ("launchctl", "bootstrap", DOMAIN):
            self.loaded = True
            if self.bootstrap_ready:
                self.models = self.bootstrap_models
                self.processes = self.bootstrap_processes
            else:
                self.models = None
                self.processes = ()
            return CommandResult(0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    def fetch_models(self, url: str, timeout: float) -> tuple[str, ...] | None:
        assert url == "http://qwen.test/v1/models"
        assert timeout > 0
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
    assert ("launchctl", "bootout", DOMAIN, str(plist)) in fake.commands
    assert ("launchctl", "bootstrap", DOMAIN, str(plist)) in fake.commands


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

    assert ("launchctl", "bootstrap", DOMAIN, str(plist)) in fake.commands


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


def test_initially_unloaded_service_appearing_during_window_fails_closed(
    tmp_path: Path,
) -> None:
    plist = _write_plist(tmp_path)
    fake = FakeQwen(loaded=False, models=None, processes=())

    with pytest.raises(RuntimeError, match="initially unloaded|became active"):
        with _guard(plist, fake):
            fake.loaded = True
            fake.models = EXPECTED_QWEN_MODELS
            fake.processes = (404,)


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


def test_owned_qwen_wrapper_plist_is_safely_accepted(tmp_path: Path) -> None:
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

    def popen(command):
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
        _popen=lambda command: process,
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
        _popen=lambda command: process,
    )

    assert result == expected_exit
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
        _popen=lambda command: process,
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

    def signal_during_popen(command):
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
