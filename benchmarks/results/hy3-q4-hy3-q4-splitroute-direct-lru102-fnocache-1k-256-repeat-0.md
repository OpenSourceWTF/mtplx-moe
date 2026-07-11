```python
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Optional

# 1. prompt schema dataclasses and JSONL loader

@dataclasses.dataclass
class PromptRecord:
    prompt_id: str
    text: str
    max_tokens: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptRecord":
        return cls(
            prompt_id=str(d["prompt_id"]),
            text=str(d["text"]),
            max_tokens=int(d["max_tokens"]),
            metadata=dict(d.get("metadata", {})),
        )

def load_prompts_jsonl(path: Path) -> list[PromptRecord]:
    out: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(PromptRecord.from_dict(json
