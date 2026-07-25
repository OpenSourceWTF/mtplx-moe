from __future__ import annotations

import contextlib
import json
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from mtplx.commands import public
from mtplx.daemon_client import (
    PORT_APP_DAEMON,
    PORT_FOREIGN,
    PORT_FREE,
    PORT_MTPLX_SERVER,
    AttachChatSession,
    PortOccupant,
    classify_port_occupant,
    detect_attachable_daemon,
    fetch_daemon_health,
    find_free_port,
    port_busy_advice,
    probe_running_daemons,
    run_attach_chat,
    stop_daemon,
)


class _StubDaemonHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:  # noqa: N802 - stdlib signature
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        config = self.server.config  # type: ignore[attr-defined]
        if self.path != "/health":
            self.send_error(404)
            return
        if config.get("garbage"):
            body = b"<html>definitely not mtplx</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        payload = {
            "ok": True,
            "model": config.get("model"),
            "model_path": config.get("model_path"),
            "startup": {
                "launch_id": config.get("launch_id"),
                "pid": config.get("pid"),
                "api_key_required": bool(config.get("api_key_required")),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        request_body = (
            json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        )
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": self.path, "body": request_body, "headers": dict(self.headers)}
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def sse(payload: dict) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        )
        sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "thinking..."},
                        "finish_reason": None,
                    }
                ]
            }
        )
        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ]
            }
        )
        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "there."}, "finish_reason": None}
                ]
            }
        )
        sse(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "mtplx_stats": {
                    "decode_tok_s": 42.0,
                    "mtp_depth": 3,
                    "request_elapsed_s": 1.25,
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")


@contextlib.contextmanager
def _stub_daemon(config: dict | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubDaemonHandler)
    server.config = dict(config or {})  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _RecordingPrinter:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def print_info(self, text, *, dim=False):
        self.events.append(("info", text))

    def print_warning(self, text):
        self.events.append(("warning", text))

    def print_error(self, text):
        self.events.append(("error", text))

    def begin_assistant(self):
        self.events.append(("begin_assistant", None))

    def stream_chunk(self, text):
        self.events.append(("content", text))

    def end_assistant(self):
        self.events.append(("end_assistant", None))

    def begin_reasoning(self):
        self.events.append(("begin_reasoning", None))

    def stream_reasoning_chunk(self, text):
        self.events.append(("reasoning", text))

    def end_reasoning(self):
        self.events.append(("end_reasoning", None))

    def print_stats(self, **fields):
        self.events.append(("stats", fields))


def test_fetch_daemon_health_parses_startup_fields():
    with _stub_daemon(
        {
            "model": "mtplx-test-model",
            "model_path": "/models/example",
            "launch_id": "native-123",
            "pid": 4242,
        }
    ) as server:
        port = server.server_address[1]
        daemon = fetch_daemon_health("127.0.0.1", port)

    assert daemon is not None
    assert daemon.model == "mtplx-test-model"
    assert daemon.model_path == "/models/example"
    assert daemon.launch_id == "native-123"
    assert daemon.pid == 4242
    assert daemon.owned_by_app is True
    assert daemon.base_url == f"http://127.0.0.1:{port}"


def test_classify_port_occupant_covers_all_kinds():
    with _stub_daemon({"model": "m", "launch_id": "native-1", "pid": 1}) as server:
        app = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert app.kind == PORT_APP_DAEMON

    with _stub_daemon({"model": "m", "launch_id": None, "pid": 1}) as server:
        cli = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert cli.kind == PORT_MTPLX_SERVER

    with _stub_daemon({"garbage": True}) as server:
        foreign = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert foreign.kind == PORT_FOREIGN

    free_port = find_free_port("127.0.0.1", 49500)
    assert free_port is not None
    assert classify_port_occupant("127.0.0.1", free_port).kind == PORT_FREE


def test_port_busy_advice_is_occupant_aware():
    app = port_busy_advice(
        PortOccupant(
            kind=PORT_APP_DAEMON,
            daemon=SimpleNamespace(model="mtplx-test"),
        ),
        port=8000,
    )
    assert any("MTPLX app" in line for line in app)
    assert any("mtplx stop --port 8000" in line for line in app)

    cli = port_busy_advice(PortOccupant(kind=PORT_MTPLX_SERVER), port=8000)
    assert any("Ctrl-C" in line for line in cli)

    foreign = port_busy_advice(PortOccupant(kind=PORT_FOREIGN), port=8000)
    assert any("another app" in line for line in foreign)


def test_find_free_port_skips_bound_ports():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        bound_port = sock.getsockname()[1]
        free = find_free_port("127.0.0.1", bound_port)
    assert free is not None
    assert free != bound_port


def test_detect_attachable_daemon_honors_kill_switch(monkeypatch):
    with _stub_daemon({"model": "m", "pid": 1}) as server:
        port = server.server_address[1]
        monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
        assert detect_attachable_daemon(ports=(port,)) is None
        monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "on")
        daemon = detect_attachable_daemon(ports=(port,))
        assert daemon is not None and daemon.port == port


def test_probe_running_daemons_skips_closed_and_foreign_ports(monkeypatch):
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "on")
    closed_port = find_free_port("127.0.0.1", 49600)
    assert closed_port is not None
    with _stub_daemon({"model": "m", "pid": 7}) as healthy:
        with _stub_daemon({"garbage": True}) as garbage:
            daemons = probe_running_daemons(
                ports=(
                    closed_port,
                    garbage.server_address[1],
                    healthy.server_address[1],
                )
            )
    assert [daemon.port for daemon in daemons] == [healthy.server_address[1]]


def test_stop_daemon_terminates_with_sigterm():
    with _stub_daemon({"model": "m", "pid": 4242}) as server:
        port = server.server_address[1]
        signals: list[int] = []
        alive = {"value": True}

        def fake_kill(pid: int, sig: int) -> None:
            assert pid == 4242
            if sig == 0:
                if not alive["value"]:
                    raise ProcessLookupError
                return
            signals.append(sig)
            if sig == signal.SIGTERM:
                alive["value"] = False

        result = stop_daemon(
            "127.0.0.1", port, kill=fake_kill, sleep=lambda _s: None
        )

    assert result["ok"] is True
    assert result["signal"] == "SIGTERM"
    assert signals == [signal.SIGTERM]


def test_stop_daemon_escalates_to_sigkill_after_grace():
    with _stub_daemon({"model": "m", "pid": 4242}) as server:
        port = server.server_address[1]
        signals: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            if sig != 0:
                signals.append(sig)

        result = stop_daemon(
            "127.0.0.1",
            port,
            grace_s=0.0,
            kill=fake_kill,
            sleep=lambda _s: None,
        )

    assert result["ok"] is True
    assert result["signal"] == "SIGKILL"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_stop_daemon_refuses_free_foreign_and_pidless_ports():
    free_port = find_free_port("127.0.0.1", 49700)
    assert free_port is not None
    assert stop_daemon("127.0.0.1", free_port)["reason"] == "no_server"

    with _stub_daemon({"garbage": True}) as server:
        result = stop_daemon("127.0.0.1", server.server_address[1])
    assert result["reason"] == "not_mtplx"

    with _stub_daemon({"model": "m", "pid": None}) as server:
        result = stop_daemon("127.0.0.1", server.server_address[1])
    assert result["reason"] == "no_pid"


def test_attach_chat_session_streams_and_keeps_history():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        port = server.server_address[1]
        daemon = fetch_daemon_health("127.0.0.1", port)
        assert daemon is not None
        session = AttachChatSession(daemon, api_key="secret-key")
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        result = session.run_turn(
            "hi",
            on_content=content_chunks.append,
            on_reasoning=reasoning_chunks.append,
        )
        second = session.run_turn("again")

        requests = server.requests  # type: ignore[attr-defined]

    assert result.content == "Hello there."
    assert result.reasoning == "thinking..."
    assert result.finish_reason == "stop"
    assert result.stats is not None and result.stats["decode_tok_s"] == 42.0
    assert "".join(content_chunks) == "Hello there."
    assert "".join(reasoning_chunks) == "thinking..."
    assert second.content == "Hello there."
    assert requests[0]["headers"].get("Authorization") == "Bearer secret-key"
    assert requests[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert requests[1]["body"]["messages"][:2] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there."},
    ]
    assert requests[1]["body"]["messages"][2] == {"role": "user", "content": "again"}


def test_run_attach_chat_one_shot_prompt():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        printer = _RecordingPrinter()

        code = run_attach_chat(daemon, prompt="hi", printer=printer)

    assert code == 0
    streamed = "".join(
        str(text) for kind, text in printer.events if kind == "content"
    )
    assert streamed == "Hello there."
    stats_events = [fields for kind, fields in printer.events if kind == "stats"]
    assert stats_events and stats_events[0]["tok_s"] == 42.0


def test_run_attach_chat_repl_supports_stats_and_exit():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        printer = _RecordingPrinter()
        answers = iter(["hi", "/stats", "/exit"])

        code = run_attach_chat(
            daemon,
            printer=printer,
            input_fn=lambda _prompt: next(answers),
        )

    assert code == 0
    stats_events = [fields for kind, fields in printer.events if kind == "stats"]
    # Once after the turn, once for /stats.
    assert len(stats_events) == 2
    reasoning = "".join(
        str(text) for kind, text in printer.events if kind == "reasoning"
    )
    assert reasoning == "thinking..."


# ---------- public.py integration: guard + port auto-select -----------------


def _guard_args(**overrides):
    defaults = {
        "prompt": None,
        "yes": False,
        "_model_explicit": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_terminal_chat_attach_guard_attaches_noninteractively(monkeypatch):
    daemon = SimpleNamespace(
        model="mtplx-test-model",
        model_path="/models/example",
        port=8000,
        host="127.0.0.1",
        owned_by_app=True,
        api_key_required=False,
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: daemon
    )
    attached: list[object] = []

    def fake_attach(target_daemon, _args):
        attached.append(target_daemon)
        return 0

    monkeypatch.setattr(public, "_run_attach_chat_for_args", fake_attach)

    code = public._terminal_chat_attach_guard(
        _guard_args(), runtime_model="/models/example"
    )

    assert code == 0
    assert attached == [daemon]


def test_terminal_chat_attach_guard_rejects_different_explicit_model(monkeypatch):
    daemon = SimpleNamespace(
        model="mtplx-test-model",
        model_path="/models/example",
        port=8000,
        host="127.0.0.1",
        owned_by_app=False,
        api_key_required=False,
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: daemon
    )

    code = public._terminal_chat_attach_guard(
        _guard_args(_model_explicit=True),
        runtime_model="/models/a-very-different-model",
    )

    assert code == 2


def test_terminal_chat_attach_guard_passes_through_without_daemon(monkeypatch):
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: None
    )

    assert (
        public._terminal_chat_attach_guard(
            _guard_args(), runtime_model="/models/example"
        )
        is None
    )


def test_quickstart_autoselect_busy_port_bumps_only_foreign(monkeypatch):
    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.find_free_port", lambda _host, _start: 8010
    )
    args = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(args, target="openwebui", cli_flags=set())
    assert args.port == 8010

    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port: PortOccupant(kind=PORT_APP_DAEMON),
    )
    reused = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        reused, target="openwebui", cli_flags=set()
    )
    assert reused.port == 8000

    explicit = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        explicit, target="openwebui", cli_flags={"port"}
    )
    assert explicit.port == 8000

    terminal = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        terminal, target="terminal", cli_flags=set()
    )
    assert terminal.port == 8000


def test_daemon_runs_model_matches_path_and_public_id(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )
    daemon = SimpleNamespace(
        model="mtplx-qwen36-27b-optimized-speed",
        model_path="/somewhere/else",
    )
    assert public._daemon_runs_model(daemon, str(model_dir)) is True

    path_daemon = SimpleNamespace(model=None, model_path=str(model_dir))
    assert public._daemon_runs_model(path_daemon, str(model_dir)) is True

    other = SimpleNamespace(model="mtplx-other", model_path="/x")
    assert public._daemon_runs_model(other, str(tmp_path / "unrelated")) is False
