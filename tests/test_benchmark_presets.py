"""Tests for named benchmark preset configurations."""

from __future__ import annotations

import argparse

import pytest

from mtplx.benchmarks import presets as P


def write(tmp_path, body: str, name: str = "presets.toml"):
    path = tmp_path / name
    path.write_text(body)
    return path


def demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    P.add_preset_arguments(parser)
    parser.add_argument("--proj-quant", choices=("q8", "q4"), default=None)
    parser.add_argument("--island-layer-count", type=int, default=None)
    parser.add_argument("--memory-limit", default="112GiB")
    parser.add_argument(
        "--model", dest="models", action="append", choices=("hy3-q2", "glm52-q2")
    )
    return parser


# --------------------------------------------------------------------------
# loading and inheritance
# --------------------------------------------------------------------------


def test_flag_and_dest_spellings_are_equivalent(tmp_path) -> None:
    path = write(
        tmp_path,
        """
        [preset.a]
        proj-quant = "q4"
        [preset.b]
        proj_quant = "q4"
        """,
    )
    loaded = P.load_preset_files([path])
    assert loaded["a"].options == loaded["b"].options == {"proj_quant": "q4"}


def test_extends_resolves_base_first_and_child_wins(tmp_path) -> None:
    path = write(
        tmp_path,
        """
        [preset.base]
        proj-quant = "q4"
        memory-limit = "108GiB"
        [preset.child]
        extends = "base"
        proj-quant = "q8"
        """,
    )
    resolved = P.resolve_preset("child", P.load_preset_files([path]))
    assert resolved.options == {"proj_quant": "q8", "memory_limit": "108GiB"}
    assert resolved.chain == ("base", "child")


def test_later_files_override_earlier_ones(tmp_path) -> None:
    first = write(tmp_path, '[preset.a]\nmemory-limit = "108GiB"\n', "a.toml")
    second = write(tmp_path, '[preset.a]\nmemory-limit = "96GiB"\n', "b.toml")
    loaded = P.load_preset_files([first, second])
    assert loaded["a"].options["memory_limit"] == "96GiB"


def test_missing_files_are_skipped_not_fatal(tmp_path) -> None:
    assert P.load_preset_files([tmp_path / "nope.toml"]) == {}


def test_circular_extends_is_reported(tmp_path) -> None:
    path = write(
        tmp_path,
        """
        [preset.a]
        extends = "b"
        [preset.b]
        extends = "a"
        """,
    )
    with pytest.raises(P.PresetError, match="circular"):
        P.resolve_preset("a", P.load_preset_files([path]))


def test_unknown_preset_lists_what_is_available(tmp_path) -> None:
    path = write(tmp_path, "[preset.real]\n")
    with pytest.raises(P.PresetError, match="unknown preset 'nope'.*real"):
        P.resolve_preset("nope", P.load_preset_files([path]))


def test_extending_a_missing_preset_names_the_extender(tmp_path) -> None:
    path = write(tmp_path, '[preset.a]\nextends = "ghost"\n')
    with pytest.raises(P.PresetError, match="'a' extends unknown preset 'ghost'"):
        P.resolve_preset("a", P.load_preset_files([path]))


# --------------------------------------------------------------------------
# the point of the whole thing: precedence
# --------------------------------------------------------------------------


def test_explicit_flag_beats_the_preset(tmp_path) -> None:
    path = write(tmp_path, '[preset.champ]\nproj-quant = "q4"\nmemory-limit = "108GiB"\n')
    parser = demo_parser()
    P.apply_preset_defaults(
        parser, P.resolve_preset("champ", P.load_preset_files([path])), environ={}
    )
    args = parser.parse_args(["--proj-quant", "q8"])
    assert args.proj_quant == "q8", "typed flag must win"
    assert args.memory_limit == "108GiB", "rest of the bundle must survive"


def test_preset_beats_the_runner_default(tmp_path) -> None:
    path = write(tmp_path, '[preset.champ]\nmemory-limit = "108GiB"\n')
    parser = demo_parser()
    assert parser.parse_args([]).memory_limit == "112GiB"
    P.apply_preset_defaults(
        parser, P.resolve_preset("champ", P.load_preset_files([path])), environ={}
    )
    assert parser.parse_args([]).memory_limit == "108GiB"


# --------------------------------------------------------------------------
# validation -- a preset that silently drops a flag is worse than one that fails
# --------------------------------------------------------------------------


def test_unknown_option_is_rejected_with_the_known_list(tmp_path) -> None:
    path = write(tmp_path, '[preset.typo]\nproj-qunat = "q8"\n')
    with pytest.raises(P.PresetError, match="unknown option.*proj_qunat"):
        P.apply_preset_defaults(
            demo_parser(),
            P.resolve_preset("typo", P.load_preset_files([path])),
            environ={},
        )


def test_value_outside_choices_is_rejected(tmp_path) -> None:
    path = write(tmp_path, '[preset.bad]\nproj-quant = "q2"\n')
    with pytest.raises(P.PresetError, match="not one of"):
        P.apply_preset_defaults(
            demo_parser(),
            P.resolve_preset("bad", P.load_preset_files([path])),
            environ={},
        )


def test_append_action_choices_are_checked_per_element(tmp_path) -> None:
    """argparse validates each element of an append action, so we must too."""

    ok = write(tmp_path, '[preset.a]\nmodels = ["hy3-q2"]\n', "ok.toml")
    P.apply_preset_defaults(
        demo_parser(), P.resolve_preset("a", P.load_preset_files([ok])), environ={}
    )

    bad = write(tmp_path, '[preset.a]\nmodels = ["hy3-q2", "nope"]\n', "bad.toml")
    with pytest.raises(P.PresetError, match="'nope'"):
        P.apply_preset_defaults(
            demo_parser(),
            P.resolve_preset("a", P.load_preset_files([bad])),
            environ={},
        )


def test_toml_boolean_env_is_rejected(tmp_path) -> None:
    """TOML true would stringify to "True", which no MTPLX_* reader accepts."""

    path = write(tmp_path, "[preset.a.env]\nMTPLX_THING = true\n")
    with pytest.raises(P.PresetError, match="TOML boolean"):
        P.load_preset_files([path])


# --------------------------------------------------------------------------
# env
# --------------------------------------------------------------------------


def test_env_is_exported_but_never_clobbers_the_caller(tmp_path) -> None:
    path = write(
        tmp_path,
        """
        [preset.a.env]
        MTPLX_FROM_PRESET = "1"
        MTPLX_ALREADY_SET = "1"
        """,
    )
    environ = {"MTPLX_ALREADY_SET": "0"}
    P.apply_preset_defaults(
        demo_parser(), P.resolve_preset("a", P.load_preset_files([path])), environ=environ
    )
    assert environ["MTPLX_FROM_PRESET"] == "1"
    assert environ["MTPLX_ALREADY_SET"] == "0", "caller's env must win"


def test_env_accumulates_through_the_extends_chain(tmp_path) -> None:
    path = write(
        tmp_path,
        """
        [preset.base.env]
        A = "1"
        [preset.child]
        extends = "base"
        [preset.child.env]
        B = "2"
        """,
    )
    resolved = P.resolve_preset("child", P.load_preset_files([path]))
    assert resolved.env == {"A": "1", "B": "2"}


# --------------------------------------------------------------------------
# argv pre-scan
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv", [["--preset", "a"], ["--preset=a"]], ids=["space", "equals"]
)
def test_preselect_accepts_both_argv_forms(tmp_path, argv) -> None:
    path = write(tmp_path, '[preset.a]\nmemory-limit = "108GiB"\n')
    parser = demo_parser()
    resolved = P.preselect_preset(
        parser, [*argv, "--presets-file", str(path)], environ={}
    )
    assert resolved is not None and resolved.name == "a"
    assert parser.parse_args([]).memory_limit == "108GiB"


def test_preselect_returns_none_without_a_preset(tmp_path) -> None:
    assert P.preselect_preset(demo_parser(), [], environ={}) is None


def test_list_presets_exits_zero(tmp_path, capsys) -> None:
    path = write(tmp_path, '[preset.a]\ndescription = "hello"\n')
    with pytest.raises(SystemExit) as exit_info:
        P.preselect_preset(
            demo_parser(), ["--list-presets", "--presets-file", str(path)], environ={}
        )
    assert exit_info.value.code == 0
    assert "hello" in capsys.readouterr().out


def test_show_preset_prints_expansion_and_exits(tmp_path, capsys) -> None:
    path = write(
        tmp_path,
        '[preset.base]\nmemory-limit = "108GiB"\n[preset.a]\nextends = "base"\nproj-quant = "q4"\n',
    )
    with pytest.raises(SystemExit) as exit_info:
        P.preselect_preset(
            demo_parser(),
            ["--show-preset", "a", "--presets-file", str(path)],
            environ={},
        )
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--proj-quant" in out and "--memory-limit" in out
    assert "base -> a" in out


# --------------------------------------------------------------------------
# the shipped file must actually work against the real runner
# --------------------------------------------------------------------------


def test_shipped_presets_file_parses() -> None:
    loaded = P.load_preset_files([P.REPO_PRESET_FILE])
    assert "championship" in loaded, "the 40.59 config should be a named preset"
    for name in loaded:
        P.resolve_preset(name, loaded)


def test_shipped_presets_all_apply_to_the_real_runner() -> None:
    """Every shipped preset must name only flags the runner actually has.

    This is the regression that matters: a preset referencing a renamed flag
    is a silently different experiment.
    """

    import importlib.util
    from pathlib import Path as _Path

    runner = (
        _Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("_runner_under_test", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded = P.load_preset_files([P.REPO_PRESET_FILE])
    for name in loaded:
        P.apply_preset_defaults(
            module.build_parser(), P.resolve_preset(name, loaded), environ={}
        )
