"""The release-notes page must be readable in light AND dark mode (#367).

Sparkle's update dialog renders the output of
scripts/render_release_notes.py in a WKWebView whose backing follows the
system appearance, and the same file is the hosted notes page. A template
that does not declare ``color-scheme: light dark`` plus a dark-scheme text
color renders default near-black text over the dark update dialog — the
unreadable "what's new" reported against 2.9.x.

These tests pin two contracts:

1. The page opts into both color schemes and pairs each with a readable
   body text color (dark override distinct from the light value).
2. Both release scripts render through this one module. The template used
   to live as hand-copied heredoc twins in release_macos_v1.sh and
   sparkle_rehearsal_kit.sh, and the rehearsal copy had already drifted
   (it lost the stylesheet entirely) — the drift guard keeps the single
   source of truth single.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RENDERER = REPO / "scripts" / "render_release_notes.py"


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "_render_release_notes_undertest", RENDERER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dark_block(page: str) -> str:
    match = re.search(r"@media \(prefers-color-scheme:\s*dark\)\{(.*?)\}\s*</style>", page, re.S)
    assert match, "template lost its prefers-color-scheme dark block"
    return match.group(1)


def test_page_declares_both_color_schemes():
    page = _renderer().render_page("<h1>x</h1>", "9.9.9")
    # Both the early meta hint and the CSS declaration: WKWebView reads
    # either, and the meta prevents a wrong-scheme first paint.
    assert '<meta name="color-scheme" content="light dark">' in page
    assert ":root{color-scheme:light dark}" in page


def test_dark_mode_overrides_body_text_and_code_chip():
    page = _renderer().render_page("<p>hello</p>", "9.9.9")
    dark = _dark_block(page)

    dark_color = re.search(r"body\{[^}]*color:\s*([^;}]+)", dark)
    assert dark_color, "dark block must set a body text color"

    light_body = re.search(r"body\{[^}]*?color:\s*([^;}]+)", page)
    assert light_body, "light styles must set an explicit body text color"
    assert dark_color.group(1) != light_body.group(1), (
        "dark body text must differ from the light value — identical values "
        "mean one scheme is unreadable"
    )
    # The inline-code chip is the other explicitly light color in the
    # template; without an override it stays a light-gray slab in dark mode.
    assert re.search(r"code\{[^}]*background:", dark)


def test_host_paints_the_background():
    # Sparkle draws the dialog behind the page and the browser paints the
    # canvas from color-scheme; an opaque body background would fight both.
    page = _renderer().render_page("<p>hello</p>", "9.9.9")
    assert re.search(r"body\{[^}]*background:\s*transparent", page)


def test_body_and_version_are_embedded():
    page = _renderer().render_page("<h2>Fixed</h2><p>things</p>", "2.9.9")
    assert "<title>MTPLX 2.9.9</title>" in page
    assert "<h2>Fixed</h2><p>things</p>" in page
    assert page.startswith("<!doctype html>")


def test_release_scripts_share_this_template():
    for script in ("release_macos_v1.sh", "sparkle_rehearsal_kit.sh"):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "render_release_notes.py" in text, (
            f"{script} must render notes through scripts/render_release_notes.py"
        )
        assert "import markdown" not in text, (
            f"{script} re-inlined a notes template copy; the template lives "
            "only in scripts/render_release_notes.py so the rehearsal and "
            "the release cannot drift apart again"
        )


def test_cli_converts_markdown_end_to_end(tmp_path):
    pytest.importorskip(
        "markdown",
        reason="release tools venv dependency; template contract is covered above",
    )
    source = tmp_path / "v9.9.9.md"
    source.write_text(
        "# MTPLX 9.9.9\n\nFaster `decode` for everyone.\n", encoding="utf-8"
    )
    destination = tmp_path / "v9.9.9.html"
    _renderer().main(["render_release_notes.py", str(source), str(destination), "9.9.9"])
    page = destination.read_text(encoding="utf-8")
    assert "<h1>MTPLX 9.9.9</h1>" in page
    assert "<code>decode</code>" in page
    assert ":root{color-scheme:light dark}" in page
