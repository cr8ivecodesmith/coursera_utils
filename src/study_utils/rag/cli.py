"""Command-line entry points for the Study RAG workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import config as config_mod


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study rag",
        description="Manage Study RAG configuration and resources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config",
        help="Manage Study RAG configuration files.",
    )
    _build_config_subcommands(config_parser)

    return parser


def _build_config_subcommands(parent: argparse.ArgumentParser) -> None:
    subparsers = parent.add_subparsers(dest="config_command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Write the default configuration template.",
    )
    init_parser.add_argument(
        "--path",
        type=str,
        help=(
            "Optional destination for the config TOML (defaults to data home)."
        ),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file if present.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the active configuration file.",
    )
    validate_parser.add_argument(
        "--path",
        type=str,
        help="Path to the config TOML (defaults to resolved data home).",
    )
    validate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success output; errors still print to stderr.",
    )

    path_parser = subparsers.add_parser(
        "path",
        help="Print the resolved config path.",
    )
    path_parser.add_argument(
        "--path",
        type=str,
        help="Optional path override to resolve/normalise.",
    )


def _to_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _handle_config(args: argparse.Namespace) -> int:
    command = args.config_command
    if command == "init":
        return _handle_config_init(args)
    if command == "validate":
        return _handle_config_validate(args)
    if command == "path":
        return _handle_config_path(args)
    raise RuntimeError(f"Unhandled config command: {command}")


def _handle_config_init(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        target = config_mod.resolve_config_path(explicit_path=explicit_path)
        config_mod.write_template(target, overwrite=args.force)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    print(f"Wrote config template to {target}")
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        cfg = config_mod.load_config(explicit_path=explicit_path)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    if not args.quiet:
        print("Configuration OK")
        print(f"  data_home: {cfg.data_home}")
        provider = cfg.providers.openai
        print(f"  chat_model: {provider.chat_model}")
        print(f"  embedding_model: {provider.embedding_model}")
        print(f"  chunk_tokens: {cfg.ingestion.chunking.tokens_per_chunk}")
    return 0


def _handle_config_path(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        path = config_mod.resolve_config_path(explicit_path=explicit_path)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # pragma: no cover - argparse already handles
        return int(exc.code)

    if args.command == "config":
        return _handle_config(args)

    parser.error("Command not implemented yet.")
    return 2


def _print_error(message: str) -> None:
    sys.stderr.write(message + "\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
