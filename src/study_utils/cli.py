"""Unified CLI entry point for study utilities."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from importlib import import_module, metadata
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
        name="init",
        summary="Bootstrap the shared study-utils workspace.",
        handler=lambda argv: _run_module_command(
            "study_utils.workspace.cli",
            "main",
            "study init",
            argv,
        ),
    ),
    CommandSpec(
        name="convert-markdown",
        summary="Convert documents into Markdown outputs.",
        handler=lambda argv: _run_module_command(
            "study_utils.convert_markdown.cli",
            "main",
            "study convert-markdown",
            argv,
        ),
    ),
    CommandSpec(
        name="transcribe-video",
        summary="Transcribe MP4 videos into plain text with Whisper.",
        handler=lambda argv: _run_module_command(
            "study_utils.transcribe_video",
            "main",
            "study transcribe-video",
            argv,
        ),
    ),
    CommandSpec(
        name="markdown-to-pdf",
        summary="Convert Markdown files into a consolidated PDF.",
        handler=lambda argv: _run_module_command(
            "study_utils.markdown_to_pdf",
            "main",
            "study markdown-to-pdf",
            argv,
        ),
    ),
    CommandSpec(
        name="text-combiner",
        summary="Merge text files into a single document.",
        handler=lambda argv: _run_module_command(
            "study_utils.text_combiner",
            "main",
            "study text-combiner",
            argv,
        ),
    ),
    CommandSpec(
        name="generate-document",
        summary="Generate AI-assisted study documents from references.",
        handler=lambda argv: _run_module_command(
            "study_utils.generate_document",
            "main",
            "study generate-document",
            argv,
        ),
    ),
    CommandSpec(
        name="rag",
        summary="Interact with the retrieval-augmented study workspace.",
        handler=lambda argv: _run_module_command(
            "study_utils.rag.cli",
            "main",
            "study rag",
            argv,
        ),
    ),
    CommandSpec(
        name="quizzer",
        summary="Launch the interactive Rich quiz session.",
        is_tui=True,
        handler=lambda argv: _run_module_command(
            "study_utils.quizzer._main",
            "main",
            "study quizzer",
            argv,
        ),
    ),
)

COMMANDS: Mapping[str, CommandSpec] = {
    spec.name: spec for spec in _COMMAND_SPECS
}


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
        "Run `study list` for commands or `study help <name>` for details.",
        "",
        format_command_table(),
    ]
    return "\n".join(parts)


def _print(
    text: str, *, stream: Optional[Callable[[str], None]] = None
) -> None:
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

    spec = COMMANDS.get(head)
    if spec and spec.handler:
        return spec.handler(tail)
    if spec:
        _print(
            f"Command '{spec.name}' is not yet implemented.",
            stream=sys.stderr.write,
        )
        return 2

    _print(f"Unknown command '{head}'.", stream=sys.stderr.write)
    _print(format_command_table(), stream=sys.stderr.write)
    return 2


def _run_module_command(
    module_name: str,
    func_name: str,
    prog_name: str,
    argv: Sequence[str],
) -> int:
    module = import_module(module_name)
    target = getattr(module, func_name)
    return _invoke_main(target, prog_name, argv)


def _invoke_main(
    func: Callable[..., object], prog_name: str, argv: Sequence[str]
) -> int:
    positional_count = _positional_parameter_count(func)
    args = list(argv)
    old_argv = sys.argv
    sys.argv = [prog_name, *args]
    try:
        result = func(args) if positional_count else func()
    except SystemExit as exc:
        return _normalize_system_exit(exc)
    finally:
        sys.argv = old_argv

    return _normalize_return(result)


def _positional_parameter_count(func: Callable[..., object]) -> int:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return 0

    count = 0
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            return max(count, 1)
    return 1 if count == 1 else 0


def _normalize_return(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    return 0


def _normalize_system_exit(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        _print(code, stream=sys.stderr.write)
        return 1
    return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation guard
    raise SystemExit(main())
