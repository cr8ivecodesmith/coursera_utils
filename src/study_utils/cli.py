"""Unified CLI entry point for study utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Iterable, Mapping, Optional, Sequence


CommandHandler = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class CommandSpec:
    """Represents a study subcommand."""

    name: str
    summary: str
    handler: Optional[CommandHandler] = None
    is_tui: bool = False


_COMMAND_SPECS: Sequence[CommandSpec] = (
    CommandSpec(
        name="transcribe-video",
        summary="Transcribe MP4 videos into plain text with Whisper.",
    ),
    CommandSpec(
        name="markdown-to-pdf",
        summary="Convert Markdown files into a consolidated PDF.",
    ),
    CommandSpec(
        name="text-combiner",
        summary="Merge text files into a single document.",
    ),
    CommandSpec(
        name="generate-document",
        summary="Generate AI-assisted study documents from references.",
    ),
    CommandSpec(
        name="quizzer",
        summary="Launch the interactive quizzer TUI.",
        is_tui=True,
    ),
)

COMMANDS: Mapping[str, CommandSpec] = {spec.name: spec for spec in _COMMAND_SPECS}


def _sorted_specs() -> Iterable[CommandSpec]:
    return _COMMAND_SPECS


def _command_name_width() -> int:
    return max(len(spec.name) for spec in _sorted_specs()) if COMMANDS else 0


def format_command_table() -> str:
    """Return a formatted command table for help output."""

    width = _command_name_width()
    lines = ["Available commands:"]
    for spec in _sorted_specs():
        name = spec.name.ljust(width)
        suffix = " (TUI)" if spec.is_tui else ""
        lines.append(f"  {name}  {spec.summary}{suffix}")
    return "\n".join(lines)


def format_usage() -> str:
    """Build the top-level usage banner with command listings."""

    parts = [
        "Usage: study <command> [args...]",
        "Run `study list` to see all commands or `study help <command>` for more detail.",
        "",
        format_command_table(),
    ]
    return "\n".join(parts)


def _print(text: str, *, stream: Optional[Callable[[str], None]] = None) -> None:
    target = stream if stream is not None else sys.stdout.write
    if text:
        target(text + "\n")


def _print_usage(to: Optional[Callable[[str], None]] = None) -> None:
    _print(format_usage(), stream=to)


def _handle_list() -> int:
    _print(format_command_table())
    return 0


def _handle_version() -> int:
    try:
        version = metadata.version("study-utils")
    except metadata.PackageNotFoundError:
        version = "unknown"
    _print(version)
    return 0


def _handle_help(argv: Sequence[str]) -> int:
    if not argv:
        _print_usage()
        return 0

    command = argv[0]
    spec = COMMANDS.get(command)
    if not spec:
        _print(f"Unknown command '{command}'.", stream=sys.stderr.write)
        _print(format_command_table(), stream=sys.stderr.write)
        return 2

    _print(f"{spec.name}: {spec.summary}")
    _print("Run `study {0} --help` for CLI-specific options.".format(spec.name))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if not args:
        _print_usage()
        return 2

    head, *tail = args

    if head in ("-h", "--help"):
        _print_usage()
        return 0

    if head in ("-V", "--version"):
        return _handle_version()

    if head == "version":
        return _handle_version()

    if head == "list":
        return _handle_list()

    if head == "help":
        return _handle_help(tail)

    _print(f"Unknown command '{head}'.", stream=sys.stderr.write)
    _print(format_command_table(), stream=sys.stderr.write)
    return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation guard
    raise SystemExit(main())
