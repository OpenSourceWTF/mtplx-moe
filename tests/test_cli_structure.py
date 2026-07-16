from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN = {"mlx", "mlx_lm", "mtplx.runtime", "mtplx.server.openai"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_cli_app_help_and_parsing_are_runtime_free():
    root = Path(__file__).resolve().parents[1]
    for relative in ("mtplx/cli_app/help.py", "mtplx/cli_app/parsing.py"):
        imports = _imports(root / relative)
        forbidden = {
            name
            for name in imports
            if any(name == item or name.startswith(item + ".") for item in FORBIDDEN)
        }
        assert not forbidden


def test_product_and_model_groups_own_expected_commands():
    from mtplx.cli_app.groups.models import COMMANDS as model_commands
    from mtplx.cli_app.groups.product import COMMANDS as product_commands

    assert product_commands == (
        "hardware",
        "start",
        "setup",
        "status",
        "stop",
        "settings",
        "ask",
        "quickstart",
        "connect",
        "openwebui",
        "models",
    )
    assert model_commands == (
        "inspect",
        "forge",
        "init",
        "profiles",
        "pull",
        "list",
        "remove",
        "model",
        "config",
    )


def test_operations_and_benchmark_groups_own_expected_commands():
    from mtplx.cli_app.groups.benchmarks import COMMANDS as benchmark_commands
    from mtplx.cli_app.groups.operations import COMMANDS as operation_commands

    assert operation_commands == (
        "env",
        "doctor",
        "report",
        "profile",
        "thermal",
        "max",
        "debug",
        "metrics",
        "dashboard",
        "integrate",
    )
    assert benchmark_commands == (
        "bench-preflight",
        "inspect-model",
        "bench",
        "qa",
        "runtime-smoke",
        "probe-contract",
        "verify-ratio",
        "verify-profile",
        "verify-qmm-probe",
        "multi-qmv-probe",
        "batch-equivalence",
        "capture-commit-equivalence",
        "mtp1-greedy-gate",
        "mtp1-sampler-smoke",
        "mtp-depth-sweep",
        "mtp-chain-probe",
        "mtp-tree-probe",
        "mtp-depth-grid",
        "mtp-adaptive",
        "dflash-mlx-baseline",
        "ddtree-mlx-baseline",
        "truth-report",
        "session-bank",
    )
