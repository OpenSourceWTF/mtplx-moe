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
    def from_dict(cls, d: dict[str
