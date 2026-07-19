"""HumanEval / MBPP code-generation evaluation.

Every benchmark in this repo so far measures throughput or acceptance. None
measures whether the output is still *correct*. That matters here because the
shipping configuration quantizes aggressively — Q2 routed experts, Q4 trunk
projections — and a throughput win that silently costs pass@1 is not a win.

This module is the scoring half: load the tasks, build prompts, extract code
from a completion, execute it against the reference tests, and compute pass@k.
Getting completions from a model is the driver's job
(``scripts/code_eval_gate.py``), so this stays importable with no MLX, no
network, and no model.

## Executing model-generated code

Scoring HumanEval means running code the model wrote. That is inherently
unsafe, so it is deliberately not the default and not silent:

- every candidate runs in a **fresh subprocess**, never in-process ``exec``
- a hard wall-clock timeout, then SIGKILL of the whole process group
- ``resource`` limits on CPU time, address space, and core dumps
- an empty-ish environment and a scratch cwd that is deleted afterwards
- the caller must pass ``allow_execution=True`` explicitly; the default
  refuses, matching the ``--allow-degraded-mtp`` convention elsewhere

This is the standard HumanEval methodology and it is appropriate for
benchmarking a local model on your own machine. It is *not* a container, and
it should not be pointed at completions from an untrusted source.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "CodeTask",
    "CodeEvalError",
    "TaskResult",
    "load_humaneval",
    "load_mbpp",
    "load_tasks",
    "extract_code",
    "build_program",
    "run_candidate",
    "pass_at_k",
    "DEFAULT_TIMEOUT_S",
]

DEFAULT_TIMEOUT_S = 15.0
_DEFAULT_ADDRESS_SPACE_BYTES = 4 * 1024**3  # 4 GiB per candidate


class CodeEvalError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodeTask:
    """One benchmark problem, normalized across HumanEval and MBPP."""

    task_id: str
    suite: str  # "humaneval" | "mbpp"
    prompt: str  # what the model is shown
    test: str  # test source appended after the candidate
    entry_point: str | None = None
    setup: str = ""  # MBPP test_setup_code

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "entry_point": self.entry_point,
        }


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    status: str  # "passed" | "failed" | "timeout" | "error" | "empty"
    detail: str = ""
    seconds: float = 0.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_humaneval(path: Path | str) -> list[CodeTask]:
    """Load the canonical ``HumanEval.jsonl``.

    Fields: task_id, prompt, canonical_solution, test, entry_point. The test
    body defines ``check(candidate)`` but does not call it, so the caller
    appends the invocation — see ``build_program``.
    """

    tasks: list[CodeTask] = []
    for row in _read_jsonl(Path(path)):
        tasks.append(
            CodeTask(
                task_id=str(row["task_id"]),
                suite="humaneval",
                prompt=str(row["prompt"]),
                test=str(row["test"]),
                entry_point=str(row["entry_point"]),
            )
        )
    if not tasks:
        raise CodeEvalError(f"no HumanEval tasks parsed from {path}")
    return tasks


def load_mbpp(path: Path | str, *, sanitized: bool = False) -> list[CodeTask]:
    """Load MBPP from ``mbpp.jsonl`` or ``sanitized-mbpp.json``.

    MBPP shows the model a natural-language description plus its asserts (the
    asserts pin the expected function name, without which the task is
    underspecified). Scoring then re-runs those asserts.
    """

    path = Path(path)
    if sanitized or path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = _read_jsonl(path)

    tasks: list[CodeTask] = []
    for row in rows:
        asserts = list(row.get("test_list") or [])
        if not asserts:
            continue
        description = str(row.get("text") or row.get("prompt") or "").strip()
        prompt = (
            f"{description}\n"
            f"Your code should satisfy these tests:\n" + "\n".join(asserts) + "\n"
        )
        tasks.append(
            CodeTask(
                task_id=f"MBPP/{row['task_id']}",
                suite="mbpp",
                prompt=prompt,
                test="\n".join(asserts),
                setup=str(row.get("test_setup_code") or ""),
            )
        )
    if not tasks:
        raise CodeEvalError(f"no MBPP tasks parsed from {path}")
    return tasks


def load_tasks(suite: str, path: Path | str) -> list[CodeTask]:
    if suite == "humaneval":
        return load_humaneval(path)
    if suite == "mbpp":
        return load_mbpp(path)
    raise CodeEvalError(f"unknown suite {suite!r}; expected humaneval or mbpp")


# --------------------------------------------------------------------------
# completion -> program
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def extract_code(completion: str, *, suite: str = "humaneval") -> str:
    """Pull runnable Python out of a chat completion.

    Instruct-tuned models wrap code in markdown fences and add prose, so a raw
    completion is usually not valid Python. Prefer the first fenced block; if
    there is no fence, fall back to the raw text.

    For HumanEval the model is continuing a function signature, so a fenced
    block that redefines the whole function is *also* valid — the caller
    concatenates prompt + completion only when nothing was fenced.
    """

    match = _FENCE.search(completion)
    if match:
        return match.group(1).rstrip()
    return completion.rstrip()


def build_program(task: CodeTask, completion: str) -> str:
    """Assemble the full source to execute: candidate + tests + invocation."""

    code = extract_code(completion, suite=task.suite)

    if task.suite == "humaneval":
        # If the model re-emitted the signature, the block stands alone;
        # otherwise it is a body continuing task.prompt.
        standalone = re.search(
            rf"def\s+{re.escape(task.entry_point or '')}\s*\(", code
        )
        body = code if standalone else task.prompt + code
        return "\n".join(
            [body, "", task.test, "", f"check({task.entry_point})", ""]
        )

    # MBPP: the asserts reference the function by name; nothing to append.
    parts = [code, ""]
    if task.setup:
        parts += [task.setup, ""]
    parts += [task.test, ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# sandboxed execution
# --------------------------------------------------------------------------

# Resource limits are defense in depth and are applied BEST-EFFORT: the real
# containment is the subprocess boundary plus the wall-clock timeout. macOS
# rejects some of these (RLIMIT_AS in particular), and a hardening call that
# raises would otherwise turn every candidate into a spurious failure — which
# is exactly what it did before this was made defensive.
_PREAMBLE = """\
import resource as _r, os as _os, faulthandler as _fh
for _name, _val in (
    ("RLIMIT_CPU", {cpu}),
    ("RLIMIT_AS", {addr}),
    ("RLIMIT_CORE", 0),
):
    _lim = getattr(_r, _name, None)
    if _lim is None:
        continue
    try:
        _r.setrlimit(_lim, (_val, _val))
    except (ValueError, OSError):
        pass
try:
    _fh.disable()
except Exception:
    pass
"""


def run_candidate(
    task: CodeTask,
    completion: str,
    *,
    allow_execution: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    address_space_bytes: int = _DEFAULT_ADDRESS_SPACE_BYTES,
) -> TaskResult:
    """Execute one candidate against its tests in an isolated subprocess.

    ``allow_execution`` must be passed explicitly. Running model-generated
    code is the entire point of this benchmark and also its only real hazard,
    so it never happens as a side effect of calling into this module.
    """

    import time

    if not allow_execution:
        raise CodeEvalError(
            "run_candidate executes model-generated code; pass "
            "allow_execution=True to acknowledge that"
        )
    if not completion.strip():
        return TaskResult(task.task_id, False, "empty", "empty completion")

    program = _PREAMBLE.format(
        cpu=max(1, int(timeout_s)), addr=int(address_space_bytes)
    ) + build_program(task, completion)

    workdir = tempfile.mkdtemp(prefix="mtplx-codeeval-")
    started = time.monotonic()
    try:
        source = Path(workdir) / "candidate.py"
        source.write_text(program)
        try:
            completed = subprocess.run(
                [sys.executable, str(source)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                # New process group so a timeout kills grandchildren too.
                start_new_session=True,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": workdir,
                    "TMPDIR": workdir,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        except subprocess.TimeoutExpired:
            return TaskResult(
                task.task_id, False, "timeout",
                f"exceeded {timeout_s}s", time.monotonic() - started,
            )
        elapsed = time.monotonic() - started
        if completed.returncode == 0:
            return TaskResult(task.task_id, True, "passed", "", elapsed)

        stderr = completed.stderr or ""
        tail = stderr.strip().splitlines()
        detail = tail[-1] if tail else f"exit {completed.returncode}"

        # Python exits 1 for a failed assert AND for a SyntaxError, but those
        # mean different things when comparing quantization arms: unparseable
        # output is degraded generation, a failed assert is wrong reasoning.
        # Keep them separate so a regression can be attributed.
        if re.search(r"^(SyntaxError|IndentationError|TabError)", stderr, re.MULTILINE):
            status = "syntax_error"
        elif completed.returncode == 1:
            status = "failed"
        else:
            status = "error"
        return TaskResult(task.task_id, False, status, detail, elapsed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex paper (Chen et al. 2021).

    ``n`` samples drawn, ``c`` of them correct. Computed as
    ``1 - C(n-c, k) / C(n, k)`` via the numerically stable product form
    rather than factorials.
    """

    if k <= 0 or n <= 0:
        raise CodeEvalError("pass_at_k requires n > 0 and k > 0")
    if c < 0 or c > n:
        raise CodeEvalError(f"pass_at_k: c={c} outside [0, n={n}]")
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= 1.0 - k / i
    return 1.0 - product


def summarize(results: Sequence[TaskResult], *, k: int = 1) -> dict[str, Any]:
    """Aggregate sample-level results into a report.

    ``results`` may contain several samples per task (same ``task_id``).
    pass@k is computed **per task and then averaged**, which is the Codex-paper
    definition — ``pass_at_k`` is a per-task estimator and feeding it a
    corpus-wide correct count is meaningless.

    An earlier signature took an ``n=`` argument and did exactly that, which
    raised as soon as the corpus-wide pass count exceeded the sample count.
    Sample counts are now derived per task, so callers cannot get it wrong.
    """

    by_task: dict[str, list[TaskResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)

    by_status: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1

    per_task: list[float] = []
    for samples in by_task.values():
        n_samples = len(samples)
        n_correct = sum(1 for s in samples if s.passed)
        if n_samples < k:
            # Cannot estimate pass@k from fewer than k samples; skip rather
            # than silently reporting an optimistic value.
            continue
        per_task.append(pass_at_k(n_samples, n_correct, k))

    tasks = len(by_task)
    fully_passed = sum(1 for s in by_task.values() if any(r.passed for r in s))
    return {
        "tasks": tasks,
        "samples": len(results),
        "passed": fully_passed,
        f"pass@{k}": (sum(per_task) / len(per_task)) if per_task else 0.0,
        "by_status": by_status,
        "failures": [
            {"task_id": r.task_id, "status": r.status, "detail": r.detail}
            for r in results
            if not r.passed
        ][:25],
    }
