# CLI and Command Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the root parser and public handler warehouse into bounded domain modules without changing established command behavior.

**Architecture:** `mtplx.cli` remains the console compatibility entry point and composes parser groups from `mtplx.cli_app`. Help and lightweight parsing are no-MLX modules. `mtplx.commands.public` remains a compatibility re-export layer while settings, support, models/integrations, runtime/server, and benchmark domains move behind it one at a time.

**Tech Stack:** Python `argparse`; pytest snapshot/characterization tests; Ruff; existing lazy import conventions.

**Assumptions:**

- Assumes settings and experiment plans have passed — will NOT duplicate their parser or handler implementations.
- Assumes existing command names/import paths are compatibility contracts — will NOT rename or delete them in structural commits.
- Assumes one domain moves per commit — will NOT mix handler extraction with behavior changes.
- Assumes exact help text/order is intentional unless the settings design changes it — will NOT opportunistically rewrite copy during extraction.

---

## File Structure

- `mtplx/cli_app/help.py` — banners, compact/verbose/advanced help, flag rendering, topic dispatch.
- `mtplx/cli_app/parsing.py` — common types, explicit-flag recording, and root parser class.
- `mtplx/cli_app/groups/product.py` — start/setup/status/stop/settings/ask/quickstart/connect/models.
- `mtplx/cli_app/groups/models.py` — inspect/forge/pull/list/remove/model/config.
- `mtplx/cli_app/groups/operations.py` — doctor/report/profile/thermal/debug/metrics/dashboard/integrations.
- `mtplx/cli_app/groups/benchmarks.py` — bench/QA/probes/truth/session commands.
- `mtplx/commands/support.py`, `models.py`, `integrations.py`, `runtime.py`, `benchmarks.py` — extracted handler domains.
- `tests/snapshots/cli/*.txt` — deterministic public help contracts.
- `tests/test_cli_structure.py` — import and layering rules.

### Task 1: Lock Parser, Help, Dispatch, and Import Behavior

**Files:**
- Create: `tests/test_cli_behavior_lock.py`
- Create: `tests/snapshots/cli/public-help.txt`
- Create: `tests/snapshots/cli/advanced-help.txt`
- Create: `tests/snapshots/cli/start-help.txt`
- Modify: none
- Test: `tests/test_cli_behavior_lock.py`

**Security flag:** `none`

- [x] **Step 1: Add characterization snapshots and namespace checks**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.cli import build_parser, main


SNAPSHOTS = Path(__file__).with_name("snapshots") / "cli"


@pytest.mark.parametrize(
    (argv, snapshot),
    [
        (["--help"], "public-help.txt"),
        (["help", "advanced"], "advanced-help.txt"),
        (["help", "start"], "start-help.txt"),
    ],
)
def test_cli_help_snapshot(argv, snapshot, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out == (SNAPSHOTS / snapshot).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    (argv, expected),
    [
        (["start", "--profile", "turbo"], {"command": "start", "profile": "turbo"}),
        (["serve", "--port", "8010"], {"command": "serve", "port": 8010}),
        (["bench", "aime", "--quick"], {"command": "bench", "bench_action": "aime", "quick": True}),
        (["model", "architectures"], {"command": "model", "model_action": "architectures"}),
    ],
)
def test_representative_namespaces(argv, expected):
    args = build_parser().parse_args(argv)
    for name, value in expected.items():
        assert getattr(args, name) == value
```

Generate the three snapshot files once from the untouched `c5bf042` behavior,
then inspect them before running the tests:

```bash
mkdir -p tests/snapshots/cli
NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'from mtplx.cli import main; raise SystemExit(main(["--help"]))' > tests/snapshots/cli/public-help.txt
NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'from mtplx.cli import main; raise SystemExit(main(["help", "advanced"]))' > tests/snapshots/cli/advanced-help.txt
NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'from mtplx.cli import main; raise SystemExit(main(["help", "start"]))' > tests/snapshots/cli/start-help.txt
```

- [x] **Step 2: Run behavior locks**

Run: `NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_behavior_lock.py tests/test_public_cli.py tests/test_cli_parity_tools.py`

Expected: PASS.

- [x] **Step 3: Commit and update issue**

```bash
git add tests/test_cli_behavior_lock.py tests/snapshots/cli
git commit -m "test: lock CLI behavior before modularization"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 1 complete: public/advanced/start help and representative parser namespaces are behavior-locked. Commit: $(git rev-parse --short HEAD)."
```

### Task 2: Extract Help and Lightweight Parsing

**Files:**
- Create: `mtplx/cli_app/__init__.py`
- Create: `mtplx/cli_app/help.py`
- Create: `mtplx/cli_app/parsing.py`
- Create: `tests/test_cli_structure.py`
- Modify: `mtplx/cli.py`

**Security flag:** `none`

- [x] **Step 1: Add a failing layering test**

```python
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
        assert not {name for name in imports if any(name == item or name.startswith(item + ".") for item in FORBIDDEN)}
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py`

Expected: FAIL because `mtplx/cli_app/help.py` does not exist.

- [x] **Step 3: Move help symbols unchanged**

Move `PUBLIC_COMMANDS`, `ADVANCED_COMMANDS`, color/banner helpers,
`_format_public_help`, `_format_advanced_help`, `_format_start_help`,
`_format_verbose_help`, `_format_commands_help`, `_format_flags_help`,
`_flag_entries_for_action`, `_flag_section_for_subparser`, `_print_help_topic`,
`_parser_command_names`, and `_print_unknown_command` to `cli_app/help.py`.
Pass `build_parser` into `_format_flags_help` and `_print_help_topic` to avoid a
cycle. Re-export the historical private names from `mtplx.cli` while tests and
callers migrate.

- [x] **Step 4: Move parser primitives unchanged**

Move `_FlagRecordingArgumentParser`, `_explicit_cli_flags`, `_profile_arg`,
`_comma_floats`, `_positive_int`, and `_kv_quant_arg` to
`cli_app/parsing.py`. Import/re-export them from `mtplx.cli`.

- [x] **Step 5: Verify snapshots, structure, and public imports**

Run: `NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py tests/test_cli_behavior_lock.py tests/test_public_cli.py tests/test_cli_parity_tools.py`

Expected: PASS with byte-identical snapshots.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/cli.py mtplx/cli_app tests/test_cli_structure.py
git commit -m "refactor: extract CLI help and parsing primitives"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 2 complete: no-MLX help and parser primitives extracted with byte-identical help. Commit: $(git rev-parse --short HEAD)."
```

### Task 3: Extract Product and Model Parser Groups

**Files:**
- Create: `mtplx/cli_app/groups/__init__.py`
- Create: `mtplx/cli_app/groups/product.py`
- Create: `mtplx/cli_app/groups/models.py`
- Modify: `mtplx/cli.py`
- Modify: `tests/test_cli_structure.py`

**Security flag:** `security`

- [x] **Step 1: Add failing group ownership assertions**

```python
def test_product_and_model_groups_own_expected_commands():
    from mtplx.cli_app.groups.models import COMMANDS as model_commands
    from mtplx.cli_app.groups.product import COMMANDS as product_commands

    assert product_commands == (
        "hardware", "start", "setup", "status", "stop", "settings", "ask",
        "quickstart", "connect", "openwebui", "models",
    )
    assert model_commands == (
        "inspect", "forge", "init", "profiles", "pull", "list", "remove",
        "model", "config",
    )
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py`

Expected: FAIL because group modules do not exist.

- [x] **Step 3: Extract parser registration functions**

Create `register_product_commands(sub, context)` and
`register_model_commands(sub, context)`. `context` is a frozen dataclass
containing the default model and lightweight handler callables required by the
registrations. Move parser declarations only; keep handler wrappers in
`mtplx.cli`. Export exact `COMMANDS` tuples from the test.

- [x] **Step 4: Compose groups from `build_parser`**

Replace the moved inline declarations with the two registration calls. Parser
objects and defaults must remain identical for the behavior-lock argv matrix.

- [x] **Step 5: Verify snapshots and focused parser suites**

Run: `NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py tests/test_cli_behavior_lock.py tests/test_public_cli.py tests/test_settings_cli.py tests/test_forge_cli.py tests/test_config.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/cli.py mtplx/cli_app/groups tests/test_cli_structure.py
git commit -m "refactor: extract product and model CLI groups"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 3 complete: product and model parser groups extracted without parser drift. Commit: $(git rev-parse --short HEAD)."
```

### Task 4: Extract Operations and Benchmark Parser Groups

**Files:**
- Create: `mtplx/cli_app/groups/operations.py`
- Create: `mtplx/cli_app/groups/benchmarks.py`
- Modify: `mtplx/cli.py`
- Modify: `tests/test_cli_structure.py`

**Security flag:** `security`

- [x] **Step 1: Add failing command ownership assertions**

```python
def test_operations_and_benchmark_groups_own_expected_commands():
    from mtplx.cli_app.groups.benchmarks import COMMANDS as benchmark_commands
    from mtplx.cli_app.groups.operations import COMMANDS as operation_commands

    assert operation_commands == (
        "env", "doctor", "report", "profile", "thermal", "max", "debug",
        "metrics", "dashboard", "integrate",
    )
    assert benchmark_commands == (
        "bench-preflight", "inspect-model", "bench", "qa", "runtime-smoke",
        "probe-contract", "verify-ratio", "verify-profile", "verify-qmm-probe",
        "multi-qmv-probe", "batch-equivalence", "capture-commit-equivalence",
        "mtp1-greedy-gate", "mtp1-sampler-smoke", "mtp-depth-sweep",
        "mtp-chain-probe", "mtp-tree-probe", "mtp-depth-grid", "mtp-adaptive",
        "dflash-mlx-baseline", "ddtree-mlx-baseline", "truth-report", "session-bank",
    )
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py`

Expected: FAIL because group modules do not exist.

- [x] **Step 3: Extract operations registration**

Move only the parser declarations for the exact operations tuple into
`register_operations_commands(sub, context)`. Keep lazy handler wrappers and
all command behavior unchanged.

- [x] **Step 4: Extract benchmark registration**

Move only the parser declarations for the exact benchmark tuple into
`register_benchmark_commands(sub, context)`. Move shared option helper
definitions used exclusively by benchmark parsers into that module; leave
runtime handler functions in place.

- [x] **Step 5: Verify all parser behavior**

Run: `NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_structure.py tests/test_cli_behavior_lock.py tests/test_public_cli.py tests/test_cli_parity_tools.py tests/test_benchmark_streamed_generation_cli.py tests/test_probe_mtp_draft_rank_cli.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/cli.py mtplx/cli_app/groups tests/test_cli_structure.py
git commit -m "refactor: extract operations and benchmark CLI groups"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 4 complete: operations and benchmark parser groups extracted with behavior locks green. Commit: $(git rev-parse --short HEAD)."
```

### Task 5: Extract Support and Settings Handlers

**Files:**
- Create: `mtplx/commands/support.py`
- Modify: `mtplx/commands/settings.py`
- Modify: `mtplx/commands/public.py`
- Create: `tests/test_command_module_boundaries.py`

**Security flag:** `security`

- [x] **Step 1: Add failing import-boundary tests**

```python
def test_public_reexports_support_and_settings_handlers():
    from mtplx.commands.public import cmd_debug_public, cmd_doctor, cmd_settings_public
    from mtplx.commands.settings import cmd_settings_public as settings_impl
    from mtplx.commands.support import cmd_debug_public as debug_impl
    from mtplx.commands.support import cmd_doctor as doctor_impl

    assert cmd_settings_public is settings_impl
    assert cmd_debug_public is debug_impl
    assert cmd_doctor is doctor_impl
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py`

Expected: FAIL because support handlers remain in `commands.public`.

- [x] **Step 3: Move support symbols unchanged**

Move doctor/report helpers and handlers from `_redact_secret_value` through
`cmd_doctor`, plus `cmd_stop_public`, `cmd_debug_public`, `_hotpath_boundary_report`,
and the existing support-bundle helpers they directly use, into
`commands/support.py`. Move the historical live `_parse_settings_pairs` and
`cmd_settings_public` into `commands/settings.py` beside native settings
handlers. Re-export all moved public/tested symbols from `commands.public`.

- [x] **Step 4: Audit imports and verify**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py tests/test_cli_parity_tools.py tests/test_public_cli.py tests/test_diagnostics.py tests/test_settings_cli.py`

Expected: PASS. Then run separate searches for direct imports, string names,
re-exports, tests/mocks, and docs references to every moved public symbol.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/commands tests/test_command_module_boundaries.py
git commit -m "refactor: extract support and settings handlers"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 5 complete: support and settings handlers extracted behind compatibility re-exports. Commit: $(git rev-parse --short HEAD)."
```

### Task 6: Extract Model and Integration Handlers

**Files:**
- Create: `mtplx/commands/models.py`
- Create: `mtplx/commands/integrations.py`
- Modify: `mtplx/commands/public.py`
- Modify: `tests/test_command_module_boundaries.py`

**Security flag:** `security`

- [x] **Step 1: Add failing re-export assertions**

```python
def test_public_reexports_model_and_integration_handlers():
    from mtplx.commands.integrations import cmd_integrate_public as integrate_impl
    from mtplx.commands.models import cmd_inspect_model_public as inspect_impl
    from mtplx.commands.models import cmd_model_public as model_impl
    from mtplx.commands.public import cmd_inspect_model_public, cmd_integrate_public, cmd_model_public

    assert cmd_inspect_model_public is inspect_impl
    assert cmd_model_public is model_impl
    assert cmd_integrate_public is integrate_impl
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py`

Expected: FAIL because domain modules do not exist.

- [x] **Step 3: Move model handler families unchanged**

Move model inspection/gating, pull/list/remove, architecture QA, and
`cmd_model_public` families into `commands/models.py`. Preserve helper function
names imported by tests. Keep Forge in `commands/forge.py`.

- [x] **Step 4: Move integration handler families unchanged**

Move OpenWebUI command construction, dashboard/connect/integrate handlers, and
Pi/OpenCode/Swival/Hermes configuration/launch helpers into
`commands/integrations.py`. Runtime generation and daemon startup remain in
`commands.public` until Task 7.

- [x] **Step 5: Re-export and verify**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py tests/test_model_catalog.py tests/test_public_cli.py tests/test_cli_parity_tools.py tests/test_server_openai.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/commands tests/test_command_module_boundaries.py
git commit -m "refactor: extract model and integration handlers"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 6 complete: model and integration handlers extracted behind compatibility re-exports. Commit: $(git rev-parse --short HEAD)."
```

### Task 7: Extract Runtime/Server and Benchmark Handlers

**Files:**
- Create: `mtplx/commands/runtime.py`
- Create: `mtplx/commands/benchmarks.py`
- Modify: `mtplx/commands/public.py`
- Modify: `tests/test_command_module_boundaries.py`

**Security flag:** `security`

- [x] **Step 1: Add failing re-export assertions**

```python
def test_public_reexports_runtime_and_benchmark_handlers():
    from mtplx.commands.benchmarks import cmd_bench_public as bench_impl
    from mtplx.commands.public import cmd_bench_public, cmd_run_public, cmd_serve_public
    from mtplx.commands.runtime import cmd_run_public as run_impl
    from mtplx.commands.runtime import cmd_serve_public as serve_impl

    assert cmd_bench_public is bench_impl
    assert cmd_run_public is run_impl
    assert cmd_serve_public is serve_impl
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py`

Expected: FAIL because runtime/benchmark modules do not exist.

- [x] **Step 3: Move benchmark families unchanged**

Move AIME, tune, bench run/suite/nightly/compare/reference, QA, profile, and
thermal benchmark families into `commands/benchmarks.py`. Preserve subprocess,
telemetry, HTTP/SSH, and result serialization behavior exactly. Re-export all
tested helper symbols from `commands.public` during compatibility.

- [x] **Step 4: Move runtime/server families unchanged**

Move serve/start banners and option resolution, `cmd_serve_public`, server-child
watchdogs, one-shot/run/chat generation, quickstart terminal/server flows, and
`cmd_quickstart_public` into `commands/runtime.py`. Integration-specific helpers
already moved in Task 6 are imported through explicit functions; do not create
a reverse import from integrations to runtime.

- [x] **Step 5: Re-export and run focused suites**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_command_module_boundaries.py tests/test_public_cli.py tests/test_cli_parity_tools.py tests/test_benchmark_streamed_generation_cli.py tests/test_benchmark_streamed_generation_concurrency_cli.py tests/test_daemon_client.py tests/test_server_openai.py`

Expected: PASS.

- [x] **Step 6: Run full suite before commit**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q`

Expected: exit 0. If a move changes behavior, restore the original body and
split the extraction at the dependency boundary rather than adjusting tests.

- [x] **Step 7: Commit and update issue**

```bash
git add mtplx/commands tests/test_command_module_boundaries.py
git commit -m "refactor: extract runtime and benchmark handlers"
gh issue comment 90 --repo davidtai/MTPLX --body "CLI task 7 complete: runtime/server and benchmark handler domains extracted with full-suite behavior equivalence. Commit: $(git rev-parse --short HEAD)."
```

### Task 8: Verify CLI Modularization and Reference Audit

**Files:**
- Modify: `docs/plans/2026-07-16-cli-modularization.md` (checkboxes only)
- Test: CLI/command tests and full suite

**Security flag:** `security`

- [x] **Step 1: Run focused CLI/command verification**

Run: `NO_COLOR=1 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_cli_behavior_lock.py tests/test_cli_structure.py tests/test_command_module_boundaries.py tests/test_public_cli.py tests/test_cli_parity_tools.py tests/test_settings_cli.py tests/test_lab_cli.py tests/test_forge_cli.py`

Expected: PASS with byte-identical help snapshots.

- [x] **Step 2: Audit references by category**

Run separate `rg` searches for each moved handler's direct imports/type
references, string literals, dynamic imports, re-exports, tests/mocks, and docs.
Every old public path either resolves through `commands.public` or is migrated;
no dynamic import points to a removed implementation.

- [x] **Step 3: Run Ruff and cycle check**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check mtplx/cli.py mtplx/cli_app mtplx/commands tests/test_cli_*.py tests/test_command_module_boundaries.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'import mtplx.cli; import mtplx.commands.public; print("imports-ok")'`

Expected: PASS and `imports-ok`.

- [x] **Step 4: Run stub scan and full suite**

Run: `! rg -n 'TODO|FIXME|placeholder|NotImplementedError' mtplx/cli_app mtplx/commands && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q`

Expected: exit 0 with no new skips or failures.

- [x] **Step 5: Record checkpoint**

```bash
gh issue comment 90 --repo davidtai/MTPLX --body "CLI modularization verified: help snapshots, parser/handler boundaries, compatibility imports, Ruff, cycle smoke, reference audit, and full suite pass on $(git rev-parse --short HEAD)."
```
