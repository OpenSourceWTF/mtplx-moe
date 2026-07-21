#!/usr/bin/env python3
"""EVAL 4 -- pass@k estimator unit test (validates the Tier-2 eval machinery).

WHAT IT MEASURES
  The pass@k / summarize() computation in mtplx/benchmarks/code_eval.py was fixed
  in commit 5ce0576 so pass@k is computed PER TASK and then averaged (the
  Codex-paper definition), not corpus-wide. This eval checks:
    1. pass_at_k(n, c, k) against the closed-form 1 - C(n-c,k)/C(n,k) via
       math.comb, for many (n, c, k).
    2. summarize() over a TOY corpus of TaskResult objects with multiple samples
       per task -- the reported pass@k must equal the mean of per-task analytic
       pass@k, and tasks with < k samples must be skipped.
    3. The known edge case from the bug: a corpus with more total passes than any
       single task's sample count (which used to raise c>n) now summarizes fine.

WHY ITS OWN CORRECTNESS IS VERIFIABLE
  Ground truth is a hand-computed analytic formula (math.comb) evaluated
  independently of the code under test. If the code matches the analytic values
  for every (n, c, k), the estimator is correct.

Also runs the repo's own pytest suites: tests/test_code_eval.py and
tests/test_code_eval_gate.py.

Run:  PYTHONPATH=<worktree> .venv/bin/python evals/passk_unit_test.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import update_results  # noqa: E402

from mtplx.benchmarks.code_eval import TaskResult, pass_at_k, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python")


def analytic_pass_at_k(n: int, c: int, k: int) -> float:
    """Closed form: 1 - C(n-c, k) / C(n, k) (0 when n-c < k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def test_pass_at_k_scalar() -> dict:
    cases = []
    ok = True
    grid = []
    for n in range(1, 11):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                grid.append((n, c, k))
    # a few canonical anchors called out explicitly
    anchors = [(5, 0, 1), (5, 1, 1), (5, 5, 1), (10, 3, 5), (10, 1, 10), (200, 10, 1)]
    for n, c, k in anchors:
        got = pass_at_k(n, c, k)
        want = analytic_pass_at_k(n, c, k)
        match = math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
        ok = ok and match
        cases.append({"n": n, "c": c, "k": k, "code": got, "analytic": want, "match": match})
    n_grid_match = 0
    for n, c, k in grid:
        got = pass_at_k(n, c, k)
        want = analytic_pass_at_k(n, c, k)
        if math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12):
            n_grid_match += 1
        else:
            ok = False
    return {
        "anchors": cases,
        "grid_size": len(grid),
        "grid_all_match": n_grid_match == len(grid),
        "grid_matched": n_grid_match,
        "passed": ok,
    }


def _corpus(spec: dict[str, tuple[int, int]]) -> list[TaskResult]:
    """spec: task_id -> (n_samples, n_correct). Emit TaskResult list."""
    results = []
    for task_id, (n, c) in spec.items():
        for i in range(n):
            passed = i < c
            results.append(
                TaskResult(
                    task_id=task_id,
                    passed=passed,
                    status="passed" if passed else "failed",
                )
            )
    return results


def test_summarize_per_task() -> dict:
    checks = []
    ok = True

    # (A) multi-sample, multi-task: pass@2 = mean of per-task analytic pass@2.
    spec = {"t1": (5, 2), "t2": (5, 0), "t3": (4, 4)}
    for k in (1, 2, 3):
        results = _corpus(spec)
        report = summarize(results, k=k)
        per_task = []
        for (n, c) in spec.values():
            if n < k:
                continue
            per_task.append(analytic_pass_at_k(n, c, k))
        want = sum(per_task) / len(per_task) if per_task else 0.0
        got = report[f"pass@{k}"]
        match = math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
        ok = ok and match
        checks.append(
            {"case": "multi-task", "k": k, "code": got, "analytic": want, "match": match}
        )

    # (B) skip tasks with < k samples.
    spec2 = {"t1": (2, 1), "t2": (5, 3)}
    report = summarize(_corpus(spec2), k=3)  # t1 has 2 < 3 -> skipped
    want = analytic_pass_at_k(5, 3, 3)  # only t2 counts
    got = report["pass@3"]
    match = math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
    ok = ok and match
    checks.append(
        {"case": "skip-<k-samples", "k": 3, "code": got, "analytic": want, "match": match}
    )

    # (C) regression from the bug: total passes > any single task's n must NOT
    # raise (old corpus-wide code did pass_at_k(n, corpus_passes, k) -> c>n).
    spec3 = {f"t{i}": (5, 5) for i in range(30)}  # 150 total passes, n=5 each
    try:
        report = summarize(_corpus(spec3), k=1)
        raised = False
        got = report["pass@1"]
    except Exception as exc:  # noqa: BLE001
        raised = True
        got = f"RAISED: {exc}"
    want = 1.0  # every task fully passes
    match = (not raised) and math.isclose(got, want, rel_tol=1e-12)
    ok = ok and match
    checks.append(
        {
            "case": "corpus-passes>n (old bug)",
            "k": 1,
            "code": got,
            "analytic": want,
            "raised": raised,
            "match": match,
        }
    )

    return {"checks": checks, "passed": ok}


def main() -> int:
    import os

    t0 = time.time()
    scalar = test_pass_at_k_scalar()
    summ = test_summarize_per_task()
    # pytest needs a PATH for the sandboxed subprocess; pass full env.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    def run_pytest_env(target: str) -> dict:
        cmd = [str(VENV_PY), "-m", "pytest", target, "-q"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        return {
            "target": target,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "summary_tail": tail[-8:],
        }

    pytests = [
        run_pytest_env("tests/test_code_eval.py"),
        run_pytest_env("tests/test_code_eval_gate.py"),
    ]

    unit_ok = scalar["passed"] and summ["passed"]
    pytest_ok = all(p["ok"] for p in pytests)
    verdict = "PASS" if (unit_ok and pytest_ok) else "FAIL"

    result = {
        "eval": "pass@k estimator unit test",
        "scalar_pass_at_k": scalar,
        "summarize_per_task": summ,
        "repo_pytests": pytests,
        "verdict": verdict,
        "seconds": round(time.time() - t0, 1),
    }
    update_results("eval4_passk_unit_test", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
