"""Unit tests for the AIME extractor and grader.

Covers:
- single \\boxed{N}
- LaTeX-spaced variants \\boxed{\\,N\\,} and \\boxed{ N }
- multiple boxes -> last one wins (model boxed scratch then final)
- missing box -> None
- fallback prose: "the final answer is X" / "answer: X"
- over-range answers extract as ints but grade as "wrong"
- empty / non-string input -> None
"""

from __future__ import annotations

import pytest

from mtplx.benchmarks.validators.aime import extract_boxed, grade


class TestExtractBoxed:
    def test_simple_boxed(self) -> None:
        assert extract_boxed(r"The answer is \boxed{482}.") == 482

    def test_boxed_at_end_of_long_text(self) -> None:
        text = "a" * 5000 + r" so the answer is \boxed{77}."
        assert extract_boxed(text) == 77

    def test_spaced_boxed_with_thinspace(self) -> None:
        assert extract_boxed(r"hence \boxed{\,247\,}.") == 247

    def test_spaced_boxed_with_whitespace(self) -> None:
        assert extract_boxed(r"\boxed{ 247 }") == 247

    def test_multi_box_last_wins(self) -> None:
        text = (
            r"trial values: \boxed{12}, then \boxed{300}, "
            r"after refining the final answer is \boxed{482}."
        )
        assert extract_boxed(text) == 482

    def test_three_boxes_last_wins(self) -> None:
        text = r"\boxed{1} ... \boxed{2} ... \boxed{3}"
        assert extract_boxed(text) == 3

    def test_missing_box_returns_none(self) -> None:
        # No box, no fallback phrasing.
        assert extract_boxed("The answer might be 99 or 482 somewhere.") is None

    def test_fallback_final_answer_is(self) -> None:
        text = "After working through the cases, the final answer is 482."
        assert extract_boxed(text) == 482

    def test_fallback_answer_colon(self) -> None:
        text = "step 1: foo\nstep 2: bar\nanswer: 196"
        assert extract_boxed(text) == 196

    def test_fallback_the_answer_is(self) -> None:
        text = "Therefore the answer is 503."
        assert extract_boxed(text) == 503

    def test_fallback_ignores_mid_text_numbers(self) -> None:
        # We only scan the tail. A number in the middle without final-answer
        # phrasing shouldn't be returned.
        text = (
            "we computed 1234 here, then 5678 there. "
            + ("filler " * 200)
            + "final answer is 7."
        )
        assert extract_boxed(text) == 7

    def test_box_beats_fallback(self) -> None:
        # If both are present, box wins.
        text = r"the final answer is 100. \boxed{200}"
        assert extract_boxed(text) == 200

    def test_over_range_extracts_as_int(self) -> None:
        # We don't clamp; the grader handles out-of-range as wrong.
        assert extract_boxed(r"\boxed{1234}") == 1234

    def test_empty_string(self) -> None:
        assert extract_boxed("") is None

    def test_non_string(self) -> None:
        assert extract_boxed(None) is None  # type: ignore[arg-type]
        assert extract_boxed(123) is None  # type: ignore[arg-type]

    def test_boxed_with_no_digits(self) -> None:
        # \boxed{x+y} shouldn't match.
        assert extract_boxed(r"\boxed{x+y}") is None

    def test_boxed_mixed_with_other_latex(self) -> None:
        text = r"so $a/b = 7$ and \boxed{070}"
        # Leading zero preserved as int (70).
        assert extract_boxed(text) == 70


class TestGrade:
    def test_correct(self) -> None:
        assert grade(277, 277) == "correct"

    def test_wrong(self) -> None:
        assert grade(123, 277) == "wrong"

    def test_abstain(self) -> None:
        assert grade(None, 277) == "abstain"

    def test_over_range_graded_wrong(self) -> None:
        # AIME max is 999; an extracted 1487 must grade as wrong.
        assert grade(1487, 487) == "wrong"

    def test_negative_extracted_graded_wrong(self) -> None:
        assert grade(-5, 5) == "wrong"

    def test_zero_correct(self) -> None:
        assert grade(0, 0) == "correct"


class TestEndToEnd:
    @pytest.mark.parametrize(
        "text,expected,answer,status",
        [
            (r"\boxed{277}", 277, 277, "correct"),
            (r"\boxed{123}", 123, 277, "wrong"),
            ("I cannot solve this.", None, 277, "abstain"),
            (r"trying \boxed{50}, then final answer is \boxed{132}.", 132, 132, "correct"),
            ("the final answer is 196.", 196, 196, "correct"),
        ],
    )
    def test_extract_then_grade(
        self, text: str, expected: int | None, answer: int, status: str
    ) -> None:
        extracted = extract_boxed(text)
        assert extracted == expected
        assert grade(extracted, answer) == status
