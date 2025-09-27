"""CLI entry point for the document-to-Markdown converter scaffolding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import (
    CollisionPolicy,
    ConfigOverrides,
    ConvertMarkdownConfigError,
    load_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study convert-markdown",
        description=(
            "Convert supported documents (PDF, DOCX, HTML, TXT, EPUB) into "
            "Markdown outputs."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to convert into Markdown.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to a TOML config file (defaults to the workspace config "
            "directory)."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help=(
            "Override the workspace root used to resolve default output and "
            "config paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory for generated Markdown files.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        help="Limit conversion to the provided extensions (e.g. pdf docx).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files when name collisions occur.",
    )
    parser.add_argument(
        "--version-output",
        action="store_true",
        help="Version conflicting outputs using -01, -02 style suffixes.",
    )
    parser.add_argument(
        "--log-level",
        help="Set the logging level for the run (defaults to INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.overwrite and args.version_output:
        parser.error("--overwrite and --version-output are mutually exclusive.")

    overrides = ConfigOverrides(
        extensions=args.extensions,
        output_dir=args.output_dir,
        collision=_collision_from_args(args),
        log_level=args.log_level,
    )

    try:
        load_config(
            config_path=args.config,
            overrides=overrides,
            workspace_path=args.workspace,
        )
    except ConvertMarkdownConfigError as exc:
        parser.error(str(exc))

    message = (
        "convert-markdown scaffolding ready; conversion pipeline is pending "
        "implementation."
    )
    sys.stdout.write(message + "\n")
    return 0


def _collision_from_args(args: argparse.Namespace) -> CollisionPolicy | None:
    if args.overwrite:
        return CollisionPolicy.OVERWRITE
    if args.version_output:
        return CollisionPolicy.VERSION
    return None


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())
