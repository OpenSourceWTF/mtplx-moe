"""Model-owner progress heartbeat.

The generation loops and the model-work scheduler tick a monotone counter
whenever the owner thread makes real forward progress: a decode cycle, a
prefill chunk, a scheduled work item. Stream watchdogs compare successive
readings — a stream that receives nothing while this counter is frozen is
wedged (a queue-lost or deadlocked request, #86), not merely slow, because
any healthy prefill or decode ticks many times per second.

Single writer (the model-owner thread); readers only compare successive
values, so a plain int under the GIL is sufficient — no lock needed.
"""

_progress = 0


def tick() -> None:
    global _progress
    _progress += 1


def value() -> int:
    return _progress
