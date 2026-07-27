from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start-laguna-s21.sh"
README = ROOT / "README.md"


def _script_text() -> str:
    return SCRIPT.read_text()


def test_laguna_launcher_has_no_machine_specific_paths_or_secrets() -> None:
    text = _script_text()

    for forbidden in (
        "/Users/",
        "qwen36-server",
        ".worktrees/laguna-perf",
        "HF_TOKEN",
        "AUTHORIZATION",
        "API_KEY",
    ):
        assert forbidden not in text


def test_laguna_launcher_derives_repository_root_from_script() -> None:
    text = _script_text()

    assert "SCRIPT_DIR=${0:A:h}" in text
    assert "REPO_ROOT=${MTPLX_REPO_ROOT:-${SCRIPT_DIR:h}}" in text
    assert 'PYTHONPATH="$REPO_ROOT' in text


def test_laguna_launcher_allows_wrapper_safe_repository_root_override() -> None:
    environment = {
        **os.environ,
        "MTPLX_REPO_ROOT": str(ROOT),
        "MTPLX_PYTHON": "/definitely/not/an/executable/python",
    }
    completed = subprocess.run(
        ["/bin/zsh", str(SCRIPT), "--print-config"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"repository_root={ROOT}" in completed.stdout


def test_laguna_launcher_defaults_to_loopback_and_exact_checkpoint() -> None:
    text = _script_text()

    assert "MODEL=${MTPLX_LAGUNA_MODEL:-mlx-community/Laguna-S-2.1-oQ4e}" in text
    assert "HOST=${MTPLX_LAGUNA_HOST:-127.0.0.1}" in text
    assert "PORT=${MTPLX_LAGUNA_PORT:-8080}" in text
    assert '"--host" "$HOST" "--port" "$PORT"' in text


def test_laguna_launcher_enables_strict_dual_lane_flags() -> None:
    text = _script_text()

    for fragment in (
        '"--scheduler-mode" "ar_batch"',
        '"--batching-preset" "latency"',
        '"--max-active-requests" "2"',
        '"--decode-batch-max" "2"',
        '"--prefill-chunk-tokens" "1024"',
        '"--batch-wait-ms" "0"',
    ):
        assert fragment in text


def test_laguna_launcher_enables_promoted_fixed_m2_route() -> None:
    text = _script_text()

    assert "export MTPLX_LAGUNA_FIXED_M2_ROUTER=1" in text
    assert "eligible-or-stock" not in text
    assert "fallback" not in text.lower()


def test_laguna_launcher_preserves_operational_guards() -> None:
    text = _script_text()

    for fragment in (
        "lsof -nP",
        "pgrep -f",
        "vm_stat",
        "pagesize",
        "MIN_AVAIL_GIB=${MTPLX_LAGUNA_MIN_AVAIL_GIB:-60}",
        "import mtplx; print(mtplx.__file__)",
        '"$REPO_ROOT"/*',
        "/health",
        "READY_TIMEOUT_S",
        'kill -9 "$SERVER_PID"',
        "MTPLX_STREAM_HIDDEN_TOOL_GUARD_TOKENS",
        "MTPLX_STREAM_HIDDEN_TOOL_GUARD_S",
    ):
        assert fragment in text


def test_laguna_launcher_has_blackwellboy_provenance_note() -> None:
    required = (
        "Blackwellboy's operational changes were applied to make the Laguna "
        "serving path work correctly."
    )

    assert required in " ".join(_script_text().split())
    assert required in " ".join(README.read_text().split())


def test_laguna_launcher_print_config_does_not_start_or_import_server() -> None:
    environment = {
        **os.environ,
        "MTPLX_PYTHON": "/definitely/not/an/executable/python",
    }
    completed = subprocess.run(
        ["/bin/zsh", str(SCRIPT), "--print-config"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "127.0.0.1:8080" in completed.stdout
    assert "mlx-community/Laguna-S-2.1-oQ4e" in completed.stdout
    assert "MTPLX_LAGUNA_FIXED_M2_ROUTER=1" in completed.stdout
    assert "--scheduler-mode ar_batch" in completed.stdout
    assert "/definitely/not/an/executable/python" in completed.stdout


def test_laguna_launcher_returns_fixed_m2_startup_failure(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    def executable(name: str, body: str) -> Path:
        path = fake_bin / name
        path.write_text(f"#!/bin/zsh\n{body}\n")
        path.chmod(0o755)
        return path

    executable("lsof", "exit 1")
    executable("pgrep", "exit 1")
    executable(
        "vm_stat",
        "print 'Pages free: 1000000.'\n"
        "print 'Pages inactive: 0.'\n"
        "print 'Pages speculative: 0.'\n"
        "print 'Pages purgeable: 0.'",
    )
    executable("pagesize", "print 4096")
    executable("curl", "exit 1")
    fake_python = executable(
        "python",
        "if [[ \"$1\" == '-c' ]]; then\n"
        f"  print {str(ROOT / 'mtplx' / '__init__.py')!r}\n"
        "  exit 0\n"
        "fi\n"
        "exit 42",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "MTPLX_PYTHON": str(fake_python),
        "MTPLX_LAGUNA_MIN_AVAIL_GIB": "1",
        "MTPLX_LAGUNA_MEM_WAIT_S": "1",
        "MTPLX_LAGUNA_READY_TIMEOUT_S": "1",
    }

    completed = subprocess.run(
        ["/bin/zsh", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 42
    assert "server died during startup (rc=42)" in completed.stderr
    assert "ready on" not in completed.stderr


def test_readme_documents_supported_launcher_and_dual_lane_contract() -> None:
    text = README.read_text()

    assert "./scripts/start-laguna-s21.sh" in text
    for fragment in (
        "MTPLX_PYTHON",
        "MTPLX_LAGUNA_MODEL",
        "MTPLX_LAGUNA_HOST",
        "MTPLX_LAGUNA_PORT",
        "MTPLX_LAGUNA_MIN_AVAIL_GIB",
        "X-MTPLX-Client: cline",
        "X-MTPLX-Client: opensource-leaderboard",
        "non-borrowing",
        "MTPLX_LAGUNA_FIXED_M2_ROUTER",
        "--scheduler-mode serial",
    ):
        assert fragment in text
