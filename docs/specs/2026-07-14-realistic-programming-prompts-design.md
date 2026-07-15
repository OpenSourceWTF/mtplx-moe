# Realistic Programming Prompt Ladder Design

## Goal

Generate deterministic, coherent programming-agent prompts at exactly 1,024,
2,048, 4,096, 8,192, and 16,384 tokenizer tokens for MTPLX prefill
benchmarks. The prompts should resemble ordinary LLM coding context instead of
repeating one synthetic module.

## Scope

- Replace the `coding-agent` prefill filler with a deterministic programming
  repository narrative built from common code, tests, documentation,
  configuration, diagnostics, and review notes.
- Preserve an actionable final user request at the end of every prompt.
- Produce exact token counts after raw or chat-template encoding.
- Make the requested five sizes the default prefill ladder contexts.
- Record enough prompt metadata to identify and compare generated inputs.
- Retain `legacy-repeat` as an explicitly non-release-valid diagnostic mode.
- Add hardware-independent tests and update benchmark documentation.

## Non-goals

- Generating prompts by calling an LLM or downloading an external corpus.
- Matching one model family's vocabulary or tokenization statistics exactly.
- Claiming that a single prompt family represents every programming language,
  framework, or coding-agent workload.
- Changing generation, sampling, MTP verification, or model runtime behavior.

## Considered Approaches

### 1. Check in five static prompt files

This makes every byte directly inspectable, but exact token counts would only
hold for the tokenizer used to create the files. It also duplicates large
amounts of content and makes policy changes cumbersome.

### 2. Assemble a deterministic prompt from curated repository sections

Build a coherent stream from stable, hand-authored sections containing common
programming vocabulary, then size it with the active tokenizer while
preserving the final request. This keeps prompts reproducible across models and
avoids runtime network or model dependencies.

This is the selected approach.

### 3. Generate prompts dynamically with an LLM

This could increase surface variety, but introduces nondeterminism, external
dependencies, provenance concerns, and circular benchmark setup. It is not
appropriate for a reproducible benchmark fixture.

## Architecture

`mtplx.prefill_bench` will expose a focused deterministic programming-context
builder. The builder will assemble ordered sections that read like a coding
agent's repository context:

1. task and repository overview;
2. Python source modules with conventional names and identifiers;
3. tests covering normal behavior and errors;
4. configuration and example data;
5. documentation, logs, and review notes;
6. the existing coherent final implementation request.

Sections will use ordinary English and common Python vocabulary. Each section
will remain meaningful in isolation and will refer to a consistent small
project rather than unrelated fragments.

The existing exact-size flow remains authoritative:

1. encode the final request in the selected raw or chat format;
2. reserve its token budget;
3. encode enough curated context to fill the remaining budget;
4. combine context and final request;
5. re-encode through the actual prompt envelope;
6. trim only from the context head to reach the exact target;
7. fail release validity if the final request cannot be preserved.

The public `prefill-ladder` command will default to
`1024,2048,4096,8192,16384`. Callers may continue to override `--contexts`.

## Interfaces and Metadata

The existing CLI surface remains compatible:

```text
mtplx bench prefill-ladder --prompt-style coding-agent
```

No network access, seed, or new dependency is required. Prompt metadata will
retain the current hashes and exact token fields. The prompt policy version
will advance so old repeated-filler results cannot be silently compared as if
they used identical inputs.

Metadata requirements:

- requested and actual token counts are equal;
- prompt style and format are explicit;
- policy version identifies the realistic programming-context builder;
- final-request hash and preservation state are recorded;
- filler hash and filler token count are recorded;
- release validity is false whenever the final request is truncated.

## Error Handling

- Reject unsupported prompt styles and formats using the existing validation.
- Reject a final request that encodes to zero tokens.
- Mark undersized contexts as non-release-valid when the final request alone
  consumes the available budget.
- Bound context assembly deterministically and raise a clear error if a
  tokenizer cannot make progress while encoding added content.
- Do not silently fall back to repetitive or random filler.

## Testing Strategy

Hardware-independent unit tests will use the existing character tokenizer and
an additional tokenizer with multi-character token behavior where useful.
Tests will verify:

- the default contexts are exactly the requested five sizes;
- every default size produces exactly the requested encoded token count;
- the final request is preserved and remains at the tail;
- generated context includes multiple realistic artifact categories;
- content is deterministic across repeated builds;
- policy metadata and hashes are stable and internally consistent;
- custom final requests, raw formatting, and undersized contexts retain their
  existing contracts;
- `legacy-repeat` remains available and non-release-valid.

Focused CLI tests will verify the defaults and serialized result metadata.

## Rollout and Compatibility

This is a benchmark-input policy change, not a runtime migration. Existing
explicit `--contexts` commands continue to work. Results from the new prompt
policy must be labeled with the new policy version and should not be treated as
prompt-identical to historical `coding_agent_tail_v2` results.

## Failure-mode Review

### Tokenizer differences produce the wrong length — critical

Mitigation: size and validate using the active tokenizer after applying the
selected chat template. Tests exercise both raw and chat envelopes and assert
exact encoded length.

### Head trimming cuts through an artifact and makes the context confusing — critical

Mitigation: assemble more than the target from ordered, self-describing
sections and trim at a decoded boundary where possible. The preserved tail
contains the complete final task, and tests assert that several artifact types
survive at every default size.

### Larger prompts become repetitive and distort benchmark realism — critical

Mitigation: use a sufficiently broad curated section cycle with varied source,
test, documentation, configuration, and diagnostic content. Add structural
tests that reject excessive repeated-section concentration.

### One synthetic Python project is not representative of all coding — minor

Accepted limitation: the first policy targets a common Python coding-agent
workload for reproducibility. Language-specific suites can be added under a
separate prompt policy if measured needs justify them.
