from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from mtplx.version import DISPLAY_VERSION, __version__


ROOT = Path(__file__).resolve().parents[1]
BLOCK_MLX = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _BlockMLX(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (
                fullname == "mlx"
                or fullname.startswith("mlx.")
                or fullname == "mlx_lm"
                or fullname.startswith("mlx_lm.")
            ):
                raise ModuleNotFoundError(f"blocked {fullname}")
            return None

    sys.meta_path.insert(0, _BlockMLX())
    """
)


def _run_no_mlx(
    tmp_path: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    blocker = tmp_path / "blocker"
    blocker.mkdir(exist_ok=True)
    (blocker / "sitecustomize.py").write_text(BLOCK_MLX, encoding="utf-8")
    pythonpath_parts = [str(blocker), str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_parts)}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_import_mtplx_without_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-c", "import mtplx; print('ok')"])

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_version_without_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-m", "mtplx.cli", "--version"])

    assert proc.returncode == 0, proc.stderr
    assert f"mtplx {DISPLAY_VERSION} ({__version__})" in proc.stdout


def test_cli_help_without_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-m", "mtplx.cli", "--help"])

    assert proc.returncode == 0, proc.stderr
    assert "Commands" in proc.stdout
    assert "Native MTP speculative decoding" in proc.stdout
    assert "mtplx quickstart" in proc.stdout
    assert "setup" in proc.stdout
    assert "status" in proc.stdout
    assert "inspect" in proc.stdout
    assert "mtplx help advanced" in proc.stdout


def test_doctor_json_reports_missing_mlx_without_traceback(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-m", "mtplx.cli", "doctor", "--json"])

    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    mlx_info = payload["environment"]["mlx"]
    assert "blocked mlx" in mlx_info["mlx_error"]
    assert "blocked mlx_lm" in mlx_info["mlx_lm_error"]
    assert "huggingface" in payload
    assert "cache_dir" in payload["huggingface"]
    assert payload["diagnostics"]["support_matrix"]["supported"]["default_model"] == (
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    check_ids = {check["id"] for check in payload["diagnostics"]["checks"]}
    assert "resource.memory" in check_ids
    assert "resource.model_cache_disk" in check_ids
    assert "model.default_repo" in check_ids


def test_doctor_json_reports_non_git_cwd_without_raw_git_error(tmp_path: Path) -> None:
    proc = _run_no_mlx(
        tmp_path,
        ["-m", "mtplx.cli", "doctor", "--deep", "--json"],
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    env = payload["environment"]
    assert env["project_root"] == str(tmp_path.resolve())
    assert env["git_branch"] == "not a git worktree"
    assert env["git_status"] == "not a git worktree"
    assert "ERROR:" not in env["git_branch"]
    assert "ERROR:" not in env["git_status"]


def test_inspect_local_non_mtp_model_without_mlx(tmp_path: Path) -> None:
    model = tmp_path / "non-mtp-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}\n', encoding="utf-8")

    proc = _run_no_mlx(
        tmp_path,
        ["-m", "mtplx.cli", "inspect", str(model), "--json"],
    )

    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["config_exists"] is True
    assert payload["model_type"] == "llama"
    assert payload["passes_primary_gate"] is False
    assert payload["mtp"]["exists"] is False
    assert payload["compatibility"]["tier"] == "no-MTP"
    assert payload["compatibility"]["exit_code"] == 2


def test_legacy_inspect_model_form_still_works_without_mlx(tmp_path: Path) -> None:
    model = tmp_path / "non-mtp-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}\n', encoding="utf-8")

    proc = _run_no_mlx(
        tmp_path,
        ["-m", "mtplx.cli", "inspect", "model", str(model), "--json"],
    )

    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["compatibility"]["tier"] == "no-MTP"


def test_run_refuses_non_mtp_model_without_importing_mlx(tmp_path: Path) -> None:
    model = tmp_path / "non-mtp-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}\n', encoding="utf-8")

    proc = _run_no_mlx(
        tmp_path,
        ["-m", "mtplx.cli", "run", "hello", "--model", str(model), "--json"],
    )

    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["error"] == "model failed MTP primary gate"
    assert payload["model"]["compatibility"]["tier"] == "no-MTP"


def test_run_reports_uncached_hf_model_without_importing_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(
        tmp_path,
        [
            "-m",
            "mtplx.cli",
            "run",
            "hello",
            "--model",
            "mtplx/example",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ],
    )

    assert proc.returncode == 6, proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["error"] == "model is not available locally"
    assert "mtplx pull mtplx/example" in payload["detail"]


def test_run_uses_config_model_without_importing_mlx(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    cache = tmp_path / "cache"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{cache}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )

    proc = _run_no_mlx(
        tmp_path,
        ["-m", "mtplx.cli", "run", "hello", "--json"],
        env_extra={"MTPLX_CONFIG": str(config)},
    )

    assert proc.returncode == 6, proc.stderr
    assert "Traceback" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model"] == "mtplx/example"
    assert "mtplx pull mtplx/example" in payload["detail"]


def test_init_dry_run_without_mlx_does_not_write_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"

    proc = _run_no_mlx(
        tmp_path,
        [
            "-m",
            "mtplx.cli",
            "init",
            "--dry-run",
            "--json",
            "--config",
            str(config),
            "--model-dir",
            str(model_dir),
        ],
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ready_for_init"
    assert payload["dry_run"] is True
    assert payload["wrote_config"] is False
    assert payload["model"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert payload["model_dir"] == str(model_dir)
    assert payload["profile"]["name"] == "sustained"
    assert payload["hardware"]["system"]
    assert payload["commands"]["pull"].startswith("mtplx pull ")
    assert not config.exists()


def test_init_write_without_mlx_writes_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"

    proc = _run_no_mlx(
        tmp_path,
        [
            "-m",
            "mtplx.cli",
            "init",
            "--write",
            "--json",
            "--config",
            str(config),
            "--model",
            "mtplx/example",
            "--model-dir",
            str(model_dir),
            "--profile",
            "exact",
            "--thermal-control",
            "none",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["wrote_config"] is True
    assert payload["downloaded"] is False
    text = config.read_text(encoding="utf-8")
    assert 'model = "mtplx/example"' in text
    assert f'model_dir = "{model_dir}"' in text
    assert 'profile = "exact"' in text
    assert 'thermal_control = "none"' in text


def test_profiles_without_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-m", "mtplx.cli", "profiles", "--json"])

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["default"] == "sustained"
    assert [profile["name"] for profile in payload["profiles"]] == [
        "stable",
        "performance-cold",
        "sustained",
        "turbo",
        "exact",
        "max-diagnostic",
    ]


def test_max_status_without_mlx(tmp_path: Path) -> None:
    proc = _run_no_mlx(tmp_path, ["-m", "mtplx.cli", "max", "--status", "--json"])

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "detection" in payload
