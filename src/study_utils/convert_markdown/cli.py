"""CLI entry point for the document-to-Markdown converter scaffolding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from study_utils.core.logging import configure_logger

from .config import (
    CollisionPolicy,
    ConfigOverrides,
    ConvertMarkdownConfigError,
    load_config,
)
from .converter import ConverterDependencies, DependencyError
from .executor import ExecutionSummary, run_conversion


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
        load_result = load_config(
            config_path=args.config,
            overrides=overrides,
            workspace_path=args.workspace,
        )
    except ConvertMarkdownConfigError as exc:
        parser.error(str(exc))

    try:
        dependencies = _build_dependencies()
    except DependencyError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    logger, log_path = configure_logger(
        "study_utils.convert_markdown",
        log_dir=load_result.layout.path_for("logs"),
        level=load_result.config.log_level,
    )
    # Avoid duplicate log handlers when invoked repeatedly (e.g., tests).
    logger.debug("convert-markdown CLI invoked")

    summary = run_conversion(
        args.paths,
        config=load_result.config,
        dependencies=dependencies,
        logger=logger,
    )

    _print_summary(summary, log_path, load_result.config.output_dir)
    return summary.exit_code


def _build_dependencies() -> ConverterDependencies:
    """Return the default conversion dependency seams."""

    raise DependencyError(
        "Conversion backends not yet available. "
        "Install optional dependencies or provide custom seams."
    )


def _print_summary(
    summary: ExecutionSummary, log_path: Path, output_dir: Path
) -> None:
    lines = [
        "convert-markdown summary:",
        "  converted: {0}".format(summary.success_count),
        "  skipped:   {0}".format(summary.skipped_count),
        "  failed:    {0}".format(summary.failure_count),
        "  output dir: {0}".format(output_dir),
        "  log file:   {0}".format(log_path),
    ]
    sys.stdout.write("\n".join(str(line) for line in lines) + "\n")


def _collision_from_args(args: argparse.Namespace) -> CollisionPolicy | None:
    if args.overwrite:
        return CollisionPolicy.OVERWRITE
    if args.version_output:
        return CollisionPolicy.VERSION
    return None


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())
