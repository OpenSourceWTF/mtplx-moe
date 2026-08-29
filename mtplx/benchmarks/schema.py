"""Benchmark record schema for MTPLX runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mtplx.chat_encoding import encode_chat_messages, is_gemma4_tokenizer
from mtplx.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


@dataclass(frozen=True)
class PromptCase:
    id: str
    category: str
    prompt: str
    max_tokens: int = 128
    notes: str = ""
    messages: list[dict[str, str]] | None = None

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    model_path: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    draft_temperature: float | None = None
    draft_top_p: float | None = None
    draft_top_k: int | None = None
    max_tokens: int = 128
    speculative_depth: int = 0
    adaptive: bool = False


@dataclass
class BenchmarkRecord:
    prompt_id: str
    category: str
    prompt_sha256: str
    backend: str
    model_path: str
    sampler: dict[str, Any]
    max_tokens: int
    generated_tokens: int = 0
    elapsed_s: float = 0.0
    tok_s: float | None = None
    accepted_tokens: int | None = None
    acceptance_by_depth: list[float] = field(default_factory=list)
    verify_time_ms: float | None = None
    draft_time_ms: float | None = None
    peak_memory_bytes: int | None = None
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_prompt_suite(path: Path | str) -> list[PromptCase]:
    cases: list[PromptCase] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(PromptCase(**data))
    return cases


def encode_prompt_case(
    tokenizer: Any,
    case: PromptCase,
    *,
    chat_template: bool = False,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> list[int]:
    if chat_template:
        messages = case.messages or [{"role": "user", "content": case.prompt}]
        if is_gemma4_tokenizer(tokenizer):
            return encode_chat_messages(
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("Tokenizer does not expose apply_chat_template")
        kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        # Only pass a validated family level: the Qwen 3.8 template
        # raise_exceptions on unknown effort values.
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **kwargs,
                )
            )
        except ValueError as exc:
            if "chat_template is not set" not in str(exc):
                raise
            return encode_chat_messages(
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
    return list(tokenizer.encode(case.prompt))


def write_jsonl(path: Path | str, records: list[BenchmarkRecord]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def empty_record(case: PromptCase, config: BenchmarkConfig) -> BenchmarkRecord:
    return BenchmarkRecord(
        prompt_id=case.id,
        category=case.category,
        prompt_sha256=case.prompt_sha256,
        backend=config.backend,
        model_path=config.model_path,
        sampler={
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "speculative_depth": config.speculative_depth,
            "adaptive": config.adaptive,
        },
        max_tokens=min(case.max_tokens, config.max_tokens),
    )


def now_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
