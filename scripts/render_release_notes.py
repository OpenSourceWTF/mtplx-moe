#!/usr/bin/env python3
"""Render a docs/releases/*.md file into the release-notes HTML page.

Single source of truth for the page template, shared by
scripts/release_macos_v1.sh (real releases) and
scripts/sparkle_rehearsal_kit.sh (update rehearsals) so the rehearsal
exercises the exact page users see. Before this module existed the two
scripts carried hand-copied heredoc twins and drifted (the rehearsal copy
lost the stylesheet entirely).

The page is rendered in two hosts:

- Sparkle's update dialog: a WKWebView whose backing follows the system
  appearance. A page that does not declare `color-scheme: light dark`
  is treated as light-only, so dark mode showed default near-black text
  over the dark dialog — the unreadable "what's new" of issue #367.
- The hosted notes page (mtplx.com/releases/notes/): the browser paints
  the canvas per scheme from the same declaration.

So the template declares both schemes, keeps the body background
transparent (the host paints it), and pairs readable text colors with
each scheme via `prefers-color-scheme`.

Usage: render_release_notes.py <source-md> <destination-html> <version>

The CLI needs the `markdown` package (the calling scripts install it into
their release tools venv). The template itself imports clean without it;
tests/test_release_notes_template.py pins the dark-mode contract.
"""

from __future__ import annotations

import pathlib
import sys

STYLE = (
    ":root{color-scheme:light dark}"
    "body{font-family:-apple-system,system-ui,sans-serif;"
    "max-width:42em;margin:2em auto;padding:0 1em;line-height:1.55;"
    "background:transparent;color:#1d1d1f}"
    "h1,h2{line-height:1.2}"
    "li{margin:.45em 0}"
    "code{background:#f2f2f4;padding:0 .25em;border-radius:4px}"
    "table{border-collapse:collapse;margin:1em 0}"
    "th,td{border:1px solid rgba(0,0,0,.18);padding:.35em .65em}"
    "th{background:#f2f2f4;text-align:left}"
    "@media (prefers-color-scheme:dark){"
    "body{color:#f5f5f7}"
    "code{background:rgba(255,255,255,.14)}"
    "th,td{border-color:rgba(255,255,255,.24)}"
    "th{background:rgba(255,255,255,.08)}"
    "}"
)


def render_page(body: str, version: str) -> str:
    """Wrap converted-markdown HTML in the shared light+dark page skeleton."""
    return (
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>MTPLX {version}</title>\n"
        f"<style>{STYLE}</style>\n"
        f"{body}\n"
    )


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(
            "usage: render_release_notes.py <source-md> <destination-html> <version>"
        )
    # Imported here, not at module top: only the CLI conversion needs the
    # package, and the template must stay importable from the plain dev venv.
    import markdown

    source, destination, version = argv[1], argv[2], argv[3]
    body = markdown.markdown(
        pathlib.Path(source).read_text(encoding="utf-8"),
        extensions=["extra"],
    )
    pathlib.Path(destination).write_text(
        render_page(body, version), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv)
