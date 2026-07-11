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

@datacl
