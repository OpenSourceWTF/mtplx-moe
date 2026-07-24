# Realistic Programming Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic, realistic programming-agent prompts at exact 1,024 through 16,384 token sizes for MTPLX prefill benchmarks.

**Architecture:** A new focused prompt-context module owns curated repository artifacts and deterministic assembly. The existing prefill builder continues to own tokenizer-aware sizing, chat-envelope encoding, tail preservation, hashes, and release-validity metadata. CLI defaults and documentation identify the new prompt policy while preserving explicit contexts and `legacy-repeat`.

**Tech Stack:** Python 3.11, pytest, existing MTPLX tokenizer interfaces and CLI.

**Assumptions:**
- Assumes each tokenizer can encode and decode ordinary UTF-8 programming text — will NOT work with tokenizers that return no tokens or cannot decode their own IDs.
- Assumes exact size means the fully formatted model input — will NOT interpret the sizes as raw character, word, or unwrapped user-content counts.
- Assumes the requested five values replace current default contexts — will NOT remove support for explicit smaller, larger, or `--full` context lists.

---

## File Structure

- Create `mtplx/benchmarks/programming_prompts.py` — curated artifact model, common-vocabulary repository sections, deterministic assembly, and structural statistics.
- Modify `mtplx/prefill_bench.py` — use the new assembler, advance policy metadata, change default contexts, and preserve tokenizer-aware exact sizing.
- Modify `tests/test_prefill_bench.py` — exact-length, determinism, diversity, tail, raw/chat, custom-tail, and failure-path coverage.
- Modify `tests/test_public_cli.py` — serialized CLI default and policy assertions.
- Modify `docs/benchmarks.md` — public prompt-policy behavior, default sizes, reproducibility, and historical comparability notes.

### Task 1: Add the deterministic programming-context assembler

**Files:**
- Create: `mtplx/benchmarks/programming_prompts.py`
- Test: `tests/test_prefill_bench.py`

**Security flag:** none

- [ ] **Step 1: Write failing structural tests**

Add these imports and tests to `tests/test_prefill_bench.py`:

```python
from mtplx.benchmarks.programming_prompts import (
    PROGRAMMING_ARTIFACT_KINDS,
    build_programming_context,
    programming_context_stats,
)


def test_programming_context_is_deterministic_and_structurally_varied() -> None:
    first = build_programming_context(minimum_characters=80_000)
    second = build_programming_context(minimum_characters=80_000)

    assert first == second
    assert len(first) >= 80_000
    stats = programming_context_stats(first)
    assert set(stats["artifact_kinds"]) == set(PROGRAMMING_ARTIFACT_KINDS)
    assert stats["artifact_count"] >= len(PROGRAMMING_ARTIFACT_KINDS) * 2
    assert stats["largest_duplicate_count"] <= 2
    for phrase in ("def ", "class ", "pytest", "README", "pyproject.toml"):
        assert phrase in first


def test_programming_context_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="minimum_characters must be positive"):
        build_programming_context(minimum_characters=0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_prefill_bench.py -k programming_context -q`

Expected: collection fails because `mtplx.benchmarks.programming_prompts` does not exist.

- [ ] **Step 3: Implement the focused assembler**

Create `mtplx/benchmarks/programming_prompts.py` with:

```python
"""Deterministic, common-vocabulary coding-agent benchmark context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


PROGRAMMING_ARTIFACT_KINDS = (
    "source",
    "test",
    "config",
    "documentation",
    "diagnostic",
    "review",
)


@dataclass(frozen=True)
class ProgrammingArtifact:
    kind: str
    path: str
    body: str

    def render(self, cycle: int) -> str:
        return (
            f"\n\n## Repository artifact: workspace_{cycle}/{self.path}\n"
            f"Artifact type: {self.kind}.\n```text\n{self.body.rstrip()}\n```\n"
        )


def _artifacts() -> tuple[ProgrammingArtifact, ...]:
    return (
        ProgrammingArtifact("documentation", "README.md", """# Task Queue\nA small Python service accepts jobs, validates input, stores state, and writes structured logs. Keep public behavior stable and make failures explicit."""),
        ProgrammingArtifact("source", "src/task_queue/models.py", """from dataclasses import dataclass, field\nfrom typing import Any\n\n@dataclass(frozen=True)\nclass Job:\n    job_id: str\n    command: str\n    metadata: dict[str, Any] = field(default_factory=dict)"""),
        ProgrammingArtifact("source", "src/task_queue/store.py", """from collections import OrderedDict\n\nclass JobStore:\n    def __init__(self, capacity: int = 128) -> None:\n        if capacity <= 0:\n            raise ValueError(\"capacity must be positive\")\n        self.capacity = capacity\n        self._items = OrderedDict()\n\n    def get(self, key: str):\n        value = self._items.pop(key)\n        self._items[key] = value\n        return value"""),
        ProgrammingArtifact("test", "tests/test_store.py", """import pytest\nfrom task_queue.store import JobStore\n\ndef test_store_rejects_invalid_capacity():\n    with pytest.raises(ValueError, match=\"positive\"):\n        JobStore(0)"""),
        ProgrammingArtifact("config", "pyproject.toml", """[project]\nname = \"task-queue\"\nrequires-python = \">=3.11\"\n\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\naddopts = \"-q\""""),
        ProgrammingArtifact("diagnostic", "logs/failed-run.log", """INFO request accepted job_id=demo-17\nWARNING retry scheduled attempt=2 delay_ms=50\nERROR state write failed reason=temporary_io_error"""),
        ProgrammingArtifact("review", "docs/review-notes.md", """The patch must preserve insertion order, reject invalid limits, use atomic file replacement, and add a regression test for duplicate job identifiers."""),
        ProgrammingArtifact("source", "src/task_queue/codec.py", """import json\nfrom typing import Any\n\ndef encode_record(value: dict[str, Any]) -> str:\n    return json.dumps(value, sort_keys=True, separators=(\",\", \":\"))\n\ndef decode_record(raw: str) -> dict[str, Any]:\n    value = json.loads(raw)\n    if not isinstance(value, dict):\n        raise ValueError(\"record must be an object\")\n    return value"""),
        ProgrammingArtifact("test", "tests/test_codec.py", """from task_queue.codec import decode_record, encode_record\n\ndef test_codec_is_deterministic():\n    assert encode_record({\"b\": 2, \"a\": 1}) == '{\"a\":1,\"b\":2}'\n    assert decode_record('{\"ok\":true}') == {\"ok\": True}"""),
        ProgrammingArtifact("documentation", "docs/api.md", """The run command reads newline-delimited JSON, validates each object, and prints a summary. Exit code 0 means success, 2 means invalid input, and 3 means a storage failure."""),
        ProgrammingArtifact("config", "config/example.json", """{\n  \"capacity\": 128,\n  \"retry_limit\": 3,\n  \"log_level\": \"INFO\",\n  \"output_path\": \"var/jobs.jsonl\"\n}"""),
        ProgrammingArtifact("diagnostic", "docs/incident.md", """A process interruption between writing data and renaming the temporary file left stale state. The fix must flush, fsync, and replace the destination without exposing partial JSON."""),
        ProgrammingArtifact("review", "docs/acceptance.md", """Run unit tests, type checks, and the command-line smoke test. Confirm deterministic output, helpful error messages, no network access, and no changes to the public schema."""),
    )


def build_programming_context(*, minimum_characters: int) -> str:
    if minimum_characters <= 0:
        raise ValueError("minimum_characters must be positive")
    rendered: list[str] = [
        "You are reviewing a normal Python repository. Read the source, tests, "
        "configuration, documentation, and diagnostics before making a small, "
        "production-safe change. Preserve public behavior and explain errors clearly."
    ]
    artifacts = _artifacts()
    cycle = 0
    while sum(map(len, rendered)) < minimum_characters:
        artifact = artifacts[cycle % len(artifacts)]
        generation = cycle // len(artifacts)
        rendered.append(artifact.render(generation))
        cycle += 1
    return "".join(rendered)


def programming_context_stats(text: str) -> dict[str, object]:
    paths = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("## Repository artifact ")]
    kinds = [kind for kind in PROGRAMMING_ARTIFACT_KINDS if f"Artifact type: {kind}." in text]
    counts = {path: paths.count(path) for path in set(paths)}
    return {
        "artifact_count": len(paths),
        "artifact_kinds": kinds,
        "largest_duplicate_count": max(counts.values(), default=0),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
```

The generation-qualified rendered path keeps every artifact occurrence distinct
while the fixed artifact order makes output deterministic.

- [ ] **Step 4: Run tests and verify they pass**

Run: `uv run pytest tests/test_prefill_bench.py -k programming_context -q`

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mtplx/benchmarks/programming_prompts.py tests/test_prefill_bench.py
git commit -m "feat(bench): add realistic programming context assembler"
```

### Task 2: Integrate exact-size prompts and new defaults

**Files:**
- Modify: `mtplx/prefill_bench.py`
- Modify: `tests/test_prefill_bench.py`
- Modify: `tests/test_public_cli.py`

**Security flag:** none

**Does NOT cover:** Explicit `--contexts`, `--full`, custom final requests, raw prompt format, and `legacy-repeat` keep their existing selectable behavior.

- [ ] **Step 1: Write failing integration tests**

Add or update tests to assert:

```python
@pytest.mark.parametrize("context_tokens", [1024, 2048, 4096, 8192, 16384])
def test_realistic_programming_prompt_has_exact_default_size(context_tokens: int) -> None:
    tokenizer = _CharTokenizer()
    first = _prompt_build_for_context(tokenizer, context_tokens)
    second = _prompt_build_for_context(tokenizer, context_tokens)
    text = tokenizer.decode(first.token_ids)

    assert first.token_ids == second.token_ids
    assert len(first.token_ids) == context_tokens
    assert DEFAULT_FINAL_REQUEST in text
    assert text.endswith("<assistant>\n")
    assert first.metadata["prompt_policy"] == "realistic_programming_v1"
    assert first.metadata["prompt_release_valid"] is True
    assert first.metadata["prompt_artifact_kinds"] >= 4


def test_default_prefill_contexts_are_requested_programming_sizes() -> None:
    assert parse_contexts(None) == [1024, 2048, 4096, 8192, 16384]
```

Update the dry-run CLI assertion in `tests/test_public_cli.py` to expect
`realistic_programming_v1`, and add a dry-run invocation without `--contexts`
that expects `[1024, 2048, 4096, 8192, 16384]`.

- [ ] **Step 2: Run tests and verify they fail**

Run: `uv run pytest tests/test_prefill_bench.py tests/test_public_cli.py -k 'prefill and (prompt or context)' -q`

Expected: failures show old defaults, old `coding_agent_tail_v2` metadata, and missing artifact metadata.

- [ ] **Step 3: Implement integration**

In `mtplx/prefill_bench.py`:

```python
from .benchmarks.programming_prompts import (
    PROGRAMMING_ARTIFACT_KINDS,
    build_programming_context,
)

DEFAULT_CONTEXTS = (1024, 2048, 4096, 8192, 16384)
FULL_CONTEXTS = DEFAULT_CONTEXTS + (32768, 65536, 131072)
PROMPT_POLICY_VERSION = "realistic_programming_v1"
```

Replace `_model_prompt_text()` only for the `coding-agent` path with a call to
`build_programming_context(minimum_characters=max(16_384, context_tokens * 8))`.
Keep `_model_prompt_text()` for `legacy-repeat`. Add metadata:

```python
"prompt_artifact_kinds": sum(
    f"Artifact type: {kind}." in filler for kind in PROGRAMMING_ARTIFACT_KINDS
),
```

Preserve the existing re-encode loop, head-only trimming, final-tail hashes,
and release-validity behavior. Add a no-progress guard that raises
`ValueError("tokenizer made no progress while sizing programming context")`
if a larger filler produces no additional encoded tokens.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `uv run pytest tests/test_prefill_bench.py tests/test_public_cli.py -k 'prefill and (prompt or context)' -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add mtplx/prefill_bench.py tests/test_prefill_bench.py tests/test_public_cli.py
git commit -m "feat(bench): generate exact realistic programming prompts"
```

### Task 3: Document and verify the prompt policy

**Files:**
- Modify: `docs/benchmarks.md`
- Modify: `docs/plans/2026-07-14-realistic-programming-prompts.md`

**Security flag:** none

- [ ] **Step 1: Add benchmark documentation**

Add a `Realistic programming prompt ladder` section to `docs/benchmarks.md`
that documents:

```markdown
## Realistic programming prompt ladder

`mtplx bench prefill-ladder` defaults to exact formatted prompt sizes of
1,024, 2,048, 4,096, 8,192, and 16,384 tokens. The `coding-agent` policy
assembles deterministic Python repository context from source, tests,
configuration, documentation, diagnostics, and review notes, then preserves a
complete implementation request at the tail.

Exact size is measured with the active tokenizer after applying the selected
`--prompt-format` (`chat` by default). The result metadata records policy
`realistic_programming_v1`, content hashes, requested and actual token counts,
and tail preservation. Results using this policy are not prompt-identical to
historical `coding_agent_tail_v2` runs. Use explicit `--contexts` for other
sizes or `--prompt-style legacy-repeat` only for historical diagnostics.
```

- [ ] **Step 2: Run complete focused verification**

Run:

```bash
uv run pytest tests/test_prefill_bench.py tests/test_public_cli.py -q
uv run ruff check mtplx/benchmarks/programming_prompts.py mtplx/prefill_bench.py tests/test_prefill_bench.py tests/test_public_cli.py
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and `git diff --check` emits no output.

- [ ] **Step 3: Inspect representative prompt invariants**

Run a small Python command using `_CharTokenizer` or an equivalent local test tokenizer and print, for every default size: actual length, policy, tail-preserved flag, artifact-kind count, and SHA-256. Confirm all five rows have exact lengths, the new policy, preserved tails, at least four artifact kinds, and deterministic hashes across a second run.

- [ ] **Step 4: Update plan checkboxes and commit**

```bash
git add docs/benchmarks.md docs/plans/2026-07-14-realistic-programming-prompts.md
git commit -m "docs: describe realistic programming prompt ladder"
```

- [ ] **Step 5: Review, publish, and open the draft PR**

Invoke `superpowers-optimized:requesting-code-review`, address verified findings,
then invoke `superpowers-optimized:verification-before-completion` and
`superpowers-optimized:finishing-a-development-branch`. Push
`codex/realistic-programming-prompts` and open a draft PR against
`experiment/moe-pr13-pr14-stack` with the design, prompt-policy compatibility
note, test evidence, and representative exact-size invariant table.
