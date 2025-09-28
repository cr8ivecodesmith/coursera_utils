"""CLI entry point for the document-to-Markdown converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from study_utils.core.logging import configure_logger
from study_utils.core import config_templates
from study_utils.core import workspace as workspace_mod
from study_utils.core.config_templates import ConfigTemplateError
from study_utils.core.workspace import WorkspaceError

from .config import (
    CONFIG_FILENAME,
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
        epilog=(
            "Run `study convert-markdown config init` to scaffold the default "
            "convert_markdown.toml template."
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
    args_list = list(argv) if argv is not None else list(sys.argv[1:])

    if args_list[:1] == ["config"]:
        return _handle_config(args_list[1:])

    parser = _build_parser()
    args = parser.parse_args(args_list)

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


def _handle_config(argv: Sequence[str]) -> int:
    parser = _build_config_parser()
    args = parser.parse_args(argv)

    return _handle_config_init(args)


def _build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study convert-markdown config",
        description=(
            "Manage configuration files for the document-to-Markdown converter."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Write the default convert_markdown.toml template.",
    )
    init_parser.add_argument(
        "--path",
        type=Path,
        help=(
            "Destination for the config TOML (defaults to the workspace "
            "config directory)."
        ),
    )
    init_parser.add_argument(
        "--workspace",
        type=Path,
        help=(
            "Workspace root override used when resolving the default config "
            "path."
        ),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if a config already exists.",
    )
    return parser


def _handle_config_init(args: argparse.Namespace) -> int:
    try:
        target = _resolve_config_target(args)
    except WorkspaceError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    template = config_templates.get_template("convert_markdown")
    try:
        written = template.write(target, overwrite=args.force)
    except ConfigTemplateError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    sys.stdout.write(f"Wrote convert_markdown config to {written}\n")
    return 0


def _resolve_config_target(args: argparse.Namespace) -> Path:
    if args.path is not None:
        candidate = args.path.expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    layout = workspace_mod.ensure_workspace(path=args.workspace)
    return layout.path_for("config") / CONFIG_FILENAME


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())
