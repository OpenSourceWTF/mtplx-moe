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

# 1. prompt schema dataclasses and JSON
