"""No-MLX CLI help rendering and topic dispatch."""

from __future__ import annotations

import argparse
import os
import sys

from mtplx.version import DISPLAY_VERSION

PUBLIC_COMMANDS = (
    ("start", "Interactive setup → chat (model · mode · web/CLI/Pi/OpenCode/Swival)"),
    ("tune", "Find the fastest AR/D1/D2/D3 depth for this Mac"),
    ("help", "Detailed help; `help commands` / `help flags` / `help <name>`"),
    ("setup", "Prepare config and the model cache"),
    ("quickstart", "Run the local OpenAI/Anthropic server"),
    ("connect", "Copy settings for Open WebUI or Claude Code"),
    ("ask", "Ask the verified local model once"),
    ("status", "Check install, model, and integration health"),
    ("stop", "Stop the MTPLX daemon answering on a port"),
    ("settings", "Get or set live daemon settings"),
    ("inspect", "Check whether a model is MTPLX-compatible"),
    ("forge", "Forge, verify, brand, discover, and publish MTP models"),
    ("hardware", "Inspect Apple Silicon / MLX acceleration eligibility"),
    ("models", "List models in the local MTPLX cache"),
)

ADVANCED_COMMANDS = {
    "Benchmark and QA": (
        ("bench *", "Nightly gates, no-fan runs, envelope compare"),
        ("qa *", "Exactness and distribution gates"),
        ("profile *", "Dispatch, thermal, compile, and eval attribution"),
    ),
    "Support": (
        ("doctor --deep", "Deep install and integration checks"),
        ("debug bundle", "Redacted support bundle"),
        ("metrics watch", "Live server metrics view"),
    ),
    "Models": (
        ("pull", "Download a model into the cache"),
        ("forge *", "Forge and publish MTPLX-branded MTP artifacts"),
        ("models", "List local cached models"),
        ("model architectures", "Architecture support matrix"),
        ("model publish-check", "HF staging readiness"),
    ),
    "Kernel Lab": (
        ("debug hotpath", "Next verify-cycle boundary map"),
        ("runtime-smoke", "Load/inject/generate smoke"),
        ("verify-profile", "Target verify section timings"),
        ("mtp-depth-sweep", "Native-MTP depth sweep"),
    ),
}


def _color_enabled() -> bool:
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("MTPLX_NO_COLOR") not in {"1", "true", "TRUE", "yes"}
    )


def _paint(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _heading(text: str) -> str:
    return _paint(text, "1;36")


def _command(text: str) -> str:
    return _paint(text, "1;33")


def _muted(text: str) -> str:
    return _paint(text, "2")


def _command_cell(text: str, width: int) -> str:
    return _command(text) + " " * max(1, width - len(text))


def _ascii_banner() -> str:
    """Inline copy of the MTPLX ASCII banner.

    Duplicated here (rather than importing ``mtplx.ui.banner``) so the
    top-level help survives even when ``rich`` and the rest of the runtime
    stack are not installed yet.
    """

    rows = [
        "███╗   ███╗ ████████╗ ██████╗  ██╗      ██╗  ██╗",
        "████╗ ████║ ╚══██╔══╝ ██╔══██╗ ██║      ╚██╗██╔╝",
        "██╔████╔██║    ██║    ██████╔╝ ██║       ╚███╔╝ ",
        "██║╚██╔╝██║    ██║    ██╔═══╝  ██║       ██╔██╗ ",
        "██║ ╚═╝ ██║    ██║    ██║      ███████╗ ██╔╝ ██╗",
        "╚═╝     ╚═╝    ╚═╝    ╚═╝      ╚══════╝ ╚═╝  ╚═╝",
    ]
    return "\n".join("  " + _paint(line, "1;36") for line in rows)


def _shell_banner_already_shown() -> bool:
    value = os.environ.get("MTPLX_SHELL_BANNER_SHOWN") or os.environ.get("MTPLX_NO_BANNER")
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _help_banner_prefix() -> str:
    if _shell_banner_already_shown():
        return ""
    return f"{_ascii_banner()}\n\n"


def _format_public_help() -> str:
    command_lines = "\n".join(
        f"  {_command_cell(name, 12)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    version_line = _muted(f"v{DISPLAY_VERSION}  ·  Native MTP speculative decoding on Apple Silicon")
    footer = _muted(
        "more: `mtplx help <command>` · `mtplx help advanced` · `mtplx --help` · `mtplx --version`"
    )
    return f"""{_help_banner_prefix()}  {version_line}

{_heading("Commands")}
{command_lines}

{_heading("Examples")}
  mtplx start                       Interactive setup, then chat
  mtplx start --fresh               Re-run the onboarding (new model/mode/surface)
  mtplx start --max --port 8000       Sustained Max browser chat with fan boost
  mtplx start pi --port 8000           Configure Pi, then start the local server
  mtplx start opencode --port 18083    Configure OpenCode Desktop for MTPLX-owned generation
  mtplx start swival --port 18084      Print Swival generic-provider command
  mtplx start hermes --port 18085      Launch Hermes Agent against MTPLX
  mtplx quickstart --profile sustained --port 8000  API server only, no chat

  {footer}
"""


def _format_advanced_help() -> str:
    sections: list[str] = []
    for title, commands in ADVANCED_COMMANDS.items():
        sections.append(f"{_heading(title)}:")
        sections.extend(
            f"  {_command_cell(command, 28)} {summary}" for command, summary in commands
        )
    return f"""{_heading("MTPLX advanced tools")}

Usage: mtplx <command> [options]

Commands suffixed with * have subcommands. Run `mtplx help <command>` for details.
The everyday path is start first. Servers, integrations, QA, and kernels live here when needed.

""" + "\n".join(sections) + """

Examples:
  mtplx bench nightly --json --dry-run
  mtplx doctor --deep
  mtplx model architectures
  mtplx debug hotpath

Docs: README.md
"""


def _format_start_help() -> str:
    return f"""{_heading("MTPLX start")}

Interactive end-to-end setup. On first run MTPLX walks you through three
quick choices: model, runtime mode, and where to chat (browser, terminal, Pi, OpenCode, Swival, or Hermes).
On later runs it offers "same as last time?" so the chat is one keypress away.

What gets asked:
  1. Model — your configured model, the verified default, custom HF, or local
  2. Mode  — Sustained, Turbo, Sustained Max, or Burst (Stable remains available via --profile safe)
  3. Where — Web UI (default), terminal CLI, Pi, OpenCode Desktop, Swival, or Hermes

Power-user shortcuts (any of these skip the onboarding wizard):
  mtplx start --fresh                 Walk the onboarding again from scratch
  mtplx start cli                     Skip onboarding; terminal chat directly
  mtplx start pi                      Configure Pi, then serve MTPLX for Pi
  mtplx start opencode --port 18083   Configure OpenCode Desktop for MTPLX-owned generation
  mtplx start swival --port 18084     Serve MTPLX and print the Swival command
  mtplx start hermes --port 18085     Serve MTPLX and open Hermes Agent in Terminal
  mtplx start --max                   Sustained Max: long-context mode with ThermalForge fan boost
  mtplx start --profile performance-cold --max
                                      Burst: old max-fan lane, max 8K context
  mtplx start --download              Pull the verified model from HF first
  mtplx start --model /path/...       Use a specific local or HF model
  mtplx start --prompt "hi"           One-shot ask and exit (non-interactive)
  mtplx start cli --no-mtp            Use target-only AR generation

Useful controls:
  --download       Download the selected/default model if missing
  --model PATH     Use a local model folder or HF repo id
  --profile sustained
                  Use the explicit long-context memory-safe native-MTP profile
  --profile safe   Use the compatibility long-response profile
  --mtp            Use native-MTP speculative generation (default)
  --no-mtp         Use target-only AR generation; MTP can be turned back on
  --prompt TEXT    (cli) Ask once and exit instead of opening chat
  --max-tokens N   (cli) Optional response cap; default uses remaining context
  --no-stats       Hide the TPS footer
  --dry-run        Preview without loading MLX

Inside terminal chat:
  /mtp status      Show whether the next turn uses MTP or AR
  /mtp off         Switch the next turn to target-only AR generation
  /mtp on          Switch the next turn back to MTP without reloading
  /stats           Print the last response stats again
  /speed           Run a 192-token comparison sample
  /exit            Quit

Aliases:
  `web` and `openwebui` -> browser chat (same as default)
  `terminal`            -> terminal chat (same as `cli`)
  `pi`                  -> Pi coding-agent connection
  `opencode`, `oc`      -> OpenCode Desktop coding-agent connection
  `swival`, `sv`        -> Swival generic-provider connection
  `hermes`              -> Hermes Agent with terminal/file/web/browser/messaging tools
"""


def _format_verbose_help() -> str:
    """Verbose help printed by `mtplx help` (no topic).

    The bare ``mtplx`` invocation prints the compact help; ``mtplx help``
    prints this fuller version with options, more examples, and pointers to
    every help subtopic (``commands``, ``flags``, ``advanced``, ``<command>``).
    """

    public_lines = "\n".join(
        f"  {_command_cell(name, 12)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    return f"""{_help_banner_prefix()}  {_muted(f"v{DISPLAY_VERSION}  ·  Native MTP speculative decoding on Apple Silicon")}

{_heading("Overview")}

  Open the local chat in your browser, or chat in this terminal. Inference
  parameters (temperature, top-p, top-k, draft depth, max tokens) live in the
  browser sidebar and persist across sessions. The OpenAI/Anthropic-compatible
  server comes up the same way.

{_heading("Usage")}

  mtplx [options] [command] [command-options]

{_heading("Options")}

  --version        Show the installed version
  --no-color       Disable terminal colors
  -h, --help       Compact help (this view is the verbose one)

{_heading("Commands (consumer surface)")}
{public_lines}

{_heading("Examples")}

  mtplx start                       Open the local chat in your browser
  mtplx start cli                   Chat in this terminal instead
  mtplx start --download            Pull the verified model from Hugging Face
  mtplx quickstart --profile sustained --port 8000  Run the API server only
  mtplx connect openwebui           Print Open WebUI integration settings
  mtplx ask "Write a tiny FastAPI app"
  mtplx inspect Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed

{_heading("Help subtopics")}

  mtplx help commands         Every command across the consumer + advanced surface
  mtplx help flags            Every flag, grouped by command
  mtplx help advanced         Benchmarks, QA, publishing, and kernel tools
  mtplx help <command>        Detailed flags for one command (argparse view)
  mtplx help start            The start user journey

  Docs: README.md
"""


def _format_commands_help() -> str:
    """Print the full list of commands across public and advanced surfaces."""

    public_lines = "\n".join(
        f"  {_command_cell(name, 16)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    advanced_sections: list[str] = []
    for title, commands in ADVANCED_COMMANDS.items():
        advanced_sections.append(_heading(title))
        advanced_sections.extend(
            f"  {_command_cell(command, 28)} {summary}" for command, summary in commands
        )
        advanced_sections.append("")
    return f"""{_heading("MTPLX commands")}

{_heading("Consumer commands")}
{public_lines}

""" + "\n".join(advanced_sections) + f"""
  {_muted("Run `mtplx help <command>` for flags on any command above.")}
"""


def _format_flags_help(build_parser) -> str:
    """Walk argparse subparsers and print every flag under every command."""

    parser = build_parser()
    sections: list[str] = []
    sections.append(_heading("MTPLX flags"))
    sections.append("")
    sections.append("  Top-level options:")
    for action in parser._actions:
        for entry in _flag_entries_for_action(action):
            sections.append(f"    {entry}")
    sections.append("")

    for sub in parser._actions:
        if not isinstance(sub, argparse._SubParsersAction):
            continue
        for command_name, sub_parser in sorted(sub.choices.items(), key=lambda item: item[0]):
            command_section = _flag_section_for_subparser(command_name, sub_parser, depth=0)
            if command_section:
                sections.extend(command_section)
                sections.append("")

    sections.append(_muted("  Run `mtplx help <command>` for the argparse view of one command."))
    return "\n".join(sections) + "\n"


def _flag_entries_for_action(action: argparse.Action) -> list[str]:
    if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
        return []
    if not action.option_strings:
        return []
    flags = ", ".join(action.option_strings)
    metavar = ""
    if action.nargs not in (0, None) or isinstance(
        action,
        (argparse._StoreAction, argparse._AppendAction),
    ):
        if not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)):
            metavar = " " + (action.metavar or action.dest.upper())
    summary = (action.help or "").replace("\n", " ").strip()
    line = f"{flags}{metavar}"
    if summary:
        line = f"{line:<28}  {summary}"
    return [line]


def _flag_section_for_subparser(
    command_name: str,
    sub_parser: argparse.ArgumentParser,
    *,
    depth: int,
) -> list[str]:
    indent = "  " * (depth + 1)
    lines: list[str] = []
    flag_lines: list[str] = []
    nested_sections: list[list[str]] = []
    for action in sub_parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for nested_name, nested_parser in sorted(action.choices.items(), key=lambda item: item[0]):
                nested = _flag_section_for_subparser(
                    f"{command_name} {nested_name}",
                    nested_parser,
                    depth=depth + 1,
                )
                if nested:
                    nested_sections.append(nested)
            continue
        for entry in _flag_entries_for_action(action):
            flag_lines.append(f"{indent}    {entry}")
    if not flag_lines and not nested_sections:
        return []
    lines.append(f"{indent}{_command(command_name)}")
    lines.extend(flag_lines)
    for section in nested_sections:
        lines.extend(section)
    return lines


def _print_help_topic(
    topic: str | None, parser: argparse.ArgumentParser, build_parser
) -> int:
    if topic in (None, ""):
        print(_format_verbose_help())
        return 0
    if topic in ("commands", "all-commands"):
        print(_format_commands_help())
        return 0
    if topic in ("flags", "options", "all-flags"):
        print(_format_flags_help(build_parser))
        return 0
    if topic == "start":
        print(_format_start_help())
        return 0
    if topic in ("advanced", "expert", "lab"):
        print(_format_advanced_help())
        return 0
    command_names = _parser_command_names(parser)
    if topic in command_names:
        try:
            parser.parse_args([topic, "--help"])
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    print(f"Unknown help topic: {topic}\n")
    print(_format_verbose_help())
    return 2


def _parser_command_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _print_unknown_command(command: str) -> int:
    print(f"Unknown command: {_command(command)}\n")
    print("Try:")
    for name, summary in PUBLIC_COMMANDS:
        print(f"  mtplx {_command_cell(name, 10)} {summary}")
    print("\nFor the full lab surface: mtplx help advanced")
    return 2
