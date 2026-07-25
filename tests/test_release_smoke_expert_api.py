from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "release_smoke_expert_api.py"
)
SPEC = importlib.util.spec_from_file_location("release_smoke_expert_api", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_run_step_wraps_lazy_stream_timeout_and_redacts_url_secrets():
    api_key = "sk-test-secret"
    base_url = "https://user:password@example.test/v1?token=url-secret"

    class LazyFailure:
        def __iter__(self):
            raise TimeoutError(
                f"request {base_url} Authorization: Bearer {api_key}"
            )

    with pytest.raises(smoke.SmokeFailure) as exc_info:
        smoke._run_step(
            "lazy stream",
            lambda: smoke._consume_openai_stream(LazyFailure()),
            api_key=api_key,
            base_url=base_url,
        )

    detail = str(exc_info.value)
    assert "TimeoutError" in detail
    assert api_key not in detail
    assert "password" not in detail
    assert "url-secret" not in detail


def test_run_step_malformed_base_url_cannot_break_error_redaction():
    api_key = "sk-malformed-secret"
    base_url = "http://[invalid/v1"

    with pytest.raises(smoke.SmokeFailure) as exc_info:
        smoke._run_step(
            "malformed URL request",
            lambda: (_ for _ in ()).throw(
                RuntimeError(
                    f"request {base_url} Authorization: Bearer {api_key}"
                )
            ),
            api_key=api_key,
            base_url=base_url,
        )

    detail = str(exc_info.value)
    assert "RuntimeError" in detail
    assert api_key not in detail
    assert base_url not in detail


def test_cli_malformed_base_url_fails_cleanly_without_secrets():
    api_key = "sk-cli-malformed-secret"
    base_url = "http://[invalid/v1"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-url",
            base_url,
            "--model",
            "model",
            "--api-key",
            api_key,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr.startswith("release smoke failed:")
    assert "base URL validation failed" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert api_key not in completed.stderr
    assert base_url not in completed.stderr


def test_run_step_does_not_swallow_keyboard_interrupt():
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        smoke._run_step(
            "interrupt",
            interrupt,
            api_key="secret",
            base_url="http://127.0.0.1:8000/v1",
        )


@pytest.mark.parametrize("backend", ["native", "python-preadv"])
def test_profile_evidence_accepts_only_supported_backends(backend):
    health = {
        "generation_mode": "ar",
        "available_generation_modes": ["ar"],
        "expert_profile": {
            "name": "hy3-oq2e-64",
            "model_key": "hy3-expert-oq2e",
            "backend": backend,
            "generation_mode": "ar",
            "customized": False,
            "evidence_commit": "14c8b57fff358bee3da2d10968a855b955b86847",
        },
        "expert_admission": {
            "revision": smoke.EXPECTED_REVISION,
            "manifest_sha256": "a" * 64,
            "bank_sha256": smoke.EXPECTED_BANK_SHA256,
        },
        "expert_streaming": {
            "model_key": "hy3-expert-oq2e",
            "manifest_sha256": "a" * 64,
            "memory_plan": {},
            "cache_by_phase": {"decode": {}},
        },
    }

    assert smoke._profile_evidence(health)["backend"] == backend

    health["expert_profile"]["backend"] = "preadv"
    with pytest.raises(smoke.SmokeFailure, match="unsupported expert backend"):
        smoke._profile_evidence(health)


def test_timeout_argument_is_positive_and_bounded(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "model",
            "--api-key",
            "key",
            "--timeout",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        smoke._parse_args()
