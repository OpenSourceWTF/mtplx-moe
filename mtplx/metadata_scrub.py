"""Strip machine-identifying provenance from artifact metadata before publish.

Forge stamps ``mtplx_runtime.json`` with the absolute paths it read and wrote
(``forge_inputs``), plus the operator's intended Hugging Face repo. Those are
useful locally and leak a home directory once uploaded. The helpers here
normalize such values without discarding the provenance that a downstream user
actually needs (source repo, source SHA, recipe, versions).

Pure standard library so it can run anywhere a manifest can be read.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


#: Provenance keys removed outright — they name only local locations.
DROPPED_PROVENANCE_KEYS = ("intended_hf_repo",)

#: Replacement stand-in for a scrubbed absolute path.
REDACTED_PATH = "<redacted>"

#: Keys whose values are paths that should be reduced to a basename.
_PATH_KEY_RE = re.compile(r"(^|_)(path|dir|directory|file|root|location)s?$")

_HOME_PREFIX_RE = re.compile(r"^(/Users/|/home/|/var/folders/|/private/var/folders/)")
_ABSOLUTE_PATH_IN_TEXT_RE = re.compile(
    r"(?:/Users/|/home/|/private/var/folders/|/var/folders/)[^\s\"';,)]*"
)


def _looks_like_local_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith("~"):
        return True
    return value.startswith("/") or bool(_HOME_PREFIX_RE.match(value))


def scrub_path_value(value: str) -> str:
    """Reduce an absolute local path to a non-identifying stand-in.

    A path keeps its final component (``experts.bin``,
    ``hy3-q4-mlx-mtp``) because that names the artifact, not the machine.
    Everything above it is dropped.
    """

    if not _looks_like_local_path(value):
        return value
    name = Path(value.rstrip("/")).name
    return f"{REDACTED_PATH}/{name}" if name else REDACTED_PATH


def scrub_text_value(value: str) -> str:
    """Redact absolute local paths embedded inside a free-text string."""

    return _ABSOLUTE_PATH_IN_TEXT_RE.sub(
        lambda match: scrub_path_value(match.group(0)), value
    )


def _scrub_value(key: str | None, value: Any) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _scrub_value(child_key, child_value)
            for child_key, child_value in value.items()
            if child_key not in DROPPED_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_scrub_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(key, item) for item in value)
    if isinstance(value, str):
        if key is not None and _PATH_KEY_RE.search(key) and _looks_like_local_path(value):
            return scrub_path_value(value)
        if _looks_like_local_path(value):
            return scrub_path_value(value)
        return scrub_text_value(value)
    return value


def scrub_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a publish-safe copy of a runtime-metadata dict.

    - Absolute local paths (``/Users/...``, ``/home/...``, temp dirs) are cut
      down to ``<redacted>/<basename>``, wherever they appear — as a value, a
      list element, or embedded in a longer string.
    - Machine-identifying provenance keys (``intended_hf_repo``) are removed.
    - Everything else, including ``source_repo``, ``source_sha``,
      ``forge_recipe`` and version stamps, is preserved verbatim.

    The input dict is never mutated.
    """

    if not isinstance(metadata, dict):
        raise TypeError("runtime metadata must be a dict")
    return _scrub_value(None, metadata)


def runtime_metadata_leaks(metadata: Any) -> list[str]:
    """Return every absolute local path still present in ``metadata``.

    Intended as a publish-time assertion: an empty list means the payload
    carries no home-directory or temp-directory paths.
    """

    leaks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            if _looks_like_local_path(value):
                leaks.append(value)
            else:
                leaks.extend(_ABSOLUTE_PATH_IN_TEXT_RE.findall(value))

    walk(metadata)
    return leaks
