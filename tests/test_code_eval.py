"""Tests for the HumanEval / MBPP scoring harness."""

from __future__ import annotations

import json

import pytest

from mtplx.benchmarks import code_eval as ce


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_load_humaneval(tmp_path) -> None:
    path = tmp_path / "he.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n    ",
                "canonical_solution": "return a + b\n",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                "entry_point": "add",
            }
        )
        + "\n"
    )
    tasks = ce.load_humaneval(path)
    assert len(tasks) == 1
    assert tasks[0].entry_point == "add" and tasks[0].suite == "humaneval"


def test_load_mbpp_embeds_the_asserts_in_the_prompt(tmp_path) -> None:
    """MBPP is underspecified without its asserts -- they pin the function name."""

    path = tmp_path / "mbpp.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": 3,
                "text": "Write a function to add two numbers.",
                "code": "def add(a,b): return a+b",
                "test_list": ["assert add(1, 2) == 3"],
                "test_setup_code": "",
            }
        )
        + "\n"
    )
    task = ce.load_mbpp(path)[0]
    assert task.task_id == "MBPP/3"
    assert "assert add(1, 2) == 3" in task.prompt


def test_empty_dataset_is_an_error(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ce.CodeEvalError):
        ce.load_humaneval(path)


# --------------------------------------------------------------------------
# extraction -- instruct models wrap code in fences
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("```python\ndef f():\n    return 1\n```", "def f():\n    return 1"),
        ("```\ndef f():\n    return 1\n```", "def f():\n    return 1"),
        ("Sure!\n```python\ndef f():\n    return 1\n```\nHope that helps!",
         "def f():\n    return 1"),
        ("    return 1", "    return 1"),  # bare continuation, no fence
    ],
)
def test_extract_code(completion, expected) -> None:
    assert ce.extract_code(completion) == expected


def test_unterminated_fence_still_yields_code() -> None:
    """Truncated at max_tokens mid-block is common; don't score it as empty."""

    assert ce.extract_code("```python\ndef f():\n    return 1") == "def f():\n    return 1"


# --------------------------------------------------------------------------
# program assembly
# --------------------------------------------------------------------------


def _he_task() -> ce.CodeTask:
    return ce.CodeTask(
        task_id="HumanEval/0",
        suite="humaneval",
        prompt="def add(a, b):\n",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
    )


def test_bare_body_is_concatenated_onto_the_prompt() -> None:
    program = ce.build_program(_he_task(), "    return a + b")
    assert program.startswith("def add(a, b):\n")
    assert "check(add)" in program


def test_redefined_function_replaces_the_prompt_rather_than_nesting() -> None:
    """A fenced block that redefines the signature must not be indented under it."""

    program = ce.build_program(
        _he_task(), "```python\ndef add(a, b):\n    return a + b\n```"
    )
    assert program.count("def add(a, b):") == 1


# --------------------------------------------------------------------------
# execution -- the part that actually runs model output
# --------------------------------------------------------------------------


def test_execution_refuses_without_explicit_opt_in() -> None:
    with pytest.raises(ce.CodeEvalError, match="allow_execution=True"):
        ce.run_candidate(_he_task(), "    return a + b")


def test_correct_solution_passes() -> None:
    result = ce.run_candidate(_he_task(), "    return a + b", allow_execution=True)
    assert result.passed and result.status == "passed"


def test_wrong_solution_fails_without_raising() -> None:
    result = ce.run_candidate(_he_task(), "    return a * b", allow_execution=True)
    assert not result.passed and result.status in {"failed", "error"}


def test_syntax_error_is_distinguished_from_a_failed_assert() -> None:
    """Both exit 1, but unparseable output and wrong logic are different
    signals when comparing quantization arms -- keep them separable."""

    broken = ce.run_candidate(_he_task(), "    return (((", allow_execution=True)
    assert not broken.passed and broken.status == "syntax_error"

    wrong = ce.run_candidate(_he_task(), "    return a * b", allow_execution=True)
    assert not wrong.passed and wrong.status == "failed"


def test_empty_completion_is_its_own_status() -> None:
    result = ce.run_candidate(_he_task(), "   \n  ", allow_execution=True)
    assert not result.passed and result.status == "empty"


def test_infinite_loop_is_killed_by_the_timeout() -> None:
    """The whole point of the sandbox: a hung candidate must not hang the run."""

    result = ce.run_candidate(
        _he_task(), "    while True:\n        pass", allow_execution=True, timeout_s=3.0
    )
    assert not result.passed and result.status == "timeout"
    assert result.seconds < 20.0


def test_candidate_cannot_leave_a_file_behind_in_the_repo(tmp_path, monkeypatch) -> None:
    """Candidates run in a scratch cwd that is removed afterwards."""

    monkeypatch.chdir(tmp_path)
    ce.run_candidate(
        _he_task(),
        "    open('escaped.txt', 'w').write('x')\n    return a + b",
        allow_execution=True,
    )
    assert not (tmp_path / "escaped.txt").exists()


# --------------------------------------------------------------------------
# pass@k
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (1, 1, 1, 1.0),
        (1, 0, 1, 0.0),
        (10, 10, 1, 1.0),
        (10, 0, 5, 0.0),
        (5, 1, 1, 0.2),
        (10, 1, 10, 1.0),  # k == n and one correct -> certain
    ],
)
def test_pass_at_k_known_values(n, c, k, expected) -> None:
    assert ce.pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_is_monotonic_in_k() -> None:
    values = [ce.pass_at_k(20, 3, k) for k in range(1, 11)]
    assert values == sorted(values)


@pytest.mark.parametrize(("n", "c", "k"), [(0, 0, 1), (5, 6, 1), (5, -1, 1), (5, 1, 0)])
def test_pass_at_k_rejects_impossible_inputs(n, c, k) -> None:
    with pytest.raises(ce.CodeEvalError):
        ce.pass_at_k(n, c, k)


def test_summarize_counts_and_lists_failures() -> None:
    results = [
        ce.TaskResult("a", True, "passed"),
        ce.TaskResult("b", False, "failed", "assert"),
        ce.TaskResult("c", False, "timeout", "slow"),
    ]
    report = ce.summarize(results)
    assert report["tasks"] == 3 and report["passed"] == 1
    assert report["pass@1"] == pytest.approx(1 / 3)
    assert report["by_status"] == {"passed": 1, "failed": 1, "timeout": 1}
    assert {f["task_id"] for f in report["failures"]} == {"b", "c"}


# --------------------------------------------------------------------------
# summarize with multiple samples per task
# --------------------------------------------------------------------------


def test_summarize_groups_samples_by_task_for_pass_at_k() -> None:
    """pass@k is a PER-TASK estimator averaged across tasks.

    The earlier signature took n= and fed the corpus-wide pass count into
    pass_at_k, which raised as soon as that count exceeded n. Sample counts
    are now derived per task so callers cannot get it wrong.
    """

    # 2 tasks x 4 samples. Task A: 2/4 correct. Task B: 0/4.
    results = (
        [ce.TaskResult("A", i < 2, "passed" if i < 2 else "failed") for i in range(4)]
        + [ce.TaskResult("B", False, "failed") for _ in range(4)]
    )
    report = ce.summarize(results, k=1)
    assert report["tasks"] == 2 and report["samples"] == 8
    assert report["passed"] == 1, "one task had at least one correct sample"
    # pass@1 = mean(2/4, 0/4) = 0.25
    assert report["pass@1"] == pytest.approx(0.25)


def test_summarize_single_sample_per_task_is_plain_accuracy() -> None:
    results = [
        ce.TaskResult("a", True, "passed"),
        ce.TaskResult("b", False, "failed"),
        ce.TaskResult("c", False, "timeout"),
    ]
    assert ce.summarize(results, k=1)["pass@1"] == pytest.approx(1 / 3)


def test_summarize_skips_tasks_with_fewer_samples_than_k() -> None:
    """Estimating pass@5 from 2 samples would be optimistic; skip instead."""

    results = [ce.TaskResult("a", True, "passed"), ce.TaskResult("a", False, "failed")]
    assert ce.summarize(results, k=5)["pass@5"] == 0.0
