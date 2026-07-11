```python
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

# 1. prompt schema dataclasses and JSONL loader

@dataclasses.dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    max_tokens: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) ->
