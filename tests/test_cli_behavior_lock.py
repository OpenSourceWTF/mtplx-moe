from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.cli import build_parser, main


SNAPSHOTS = Path(__file__).with_name("snapshots") / "cli"


@pytest.mark.parametrize(
    ("argv", "snapshot"),
    [
        (["--help"], "public-help.txt"),
        (["help", "advanced"], "advanced-help.txt"),
        (["help", "start"], "start-help.txt"),
    ],
)
def test_cli_help_snapshot(argv, snapshot, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out == (SNAPSHOTS / snapshot).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["start", "--profile", "turbo"], {"command": "start", "profile": "turbo"}),
        (["serve", "--port", "8010"], {"command": "serve", "port": 8010}),
        (
            ["bench", "aime", "--quick"],
            {"command": "bench", "bench_action": "aime", "quick": True},
        ),
        (
            ["model", "architectures"],
            {"command": "model", "model_action": "architectures"},
        ),
        (
            ["config", "show"],
            {"command": "config", "config_action": "show"},
        ),
    ],
)
def test_representative_namespaces(argv, expected):
    args = build_parser().parse_args(argv)
    for name, value in expected.items():
        assert getattr(args, name) == value
