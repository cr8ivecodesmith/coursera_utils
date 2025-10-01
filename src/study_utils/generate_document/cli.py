"""Command-line interface for ``study generate-document``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from study_utils.core import config_templates, parse_extensions
from study_utils.core import workspace as workspace_mod
from study_utils.core.config_templates import ConfigTemplateError
from study_utils.core.workspace import WorkspaceError

from .config import CONFIG_FILENAME, find_config_path
from .runner import generate_document


DEFAULT_EXTENSIONS = {"txt", "md", "markdown"}
CONFIG_TEMPLATE_NAME = "generate_document"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study generate-document",
        description=(
            "Generate a Markdown document from reference files using an AI "
            "prompt"
        ),
        epilog=(
            "Run `study generate-document config init` to scaffold the "
            "default documents.toml template."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "DOC_TYPE",
        help="Document type key defined in documents.toml",
    )
    parser.add_argument("OUTPUT", help="Output markdown filename")
    parser.add_argument(
        "INPUTS",
        nargs="+",
        help="Reference files and/or directories to scan",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        help=(
            "File extensions to include (e.g. txt md markdown). Defaults "
            "include txt, md, markdown"
        ),
    )
    parser.add_argument(
        "--level-limit",
        type=int,
        default=0,
        help="Directory depth to traverse for directories",
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to documents.toml. Defaults to ./documents.toml then "
            "bundled study_utils/documents.toml"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(argv) if argv is not None else list(sys.argv[1:])

    if args_list[:1] == ["config"]:
        return _handle_config(args_list[1:])

    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    output_path = Path(args.OUTPUT).expanduser().resolve()
    input_paths = [Path(arg).expanduser().resolve() for arg in args.INPUTS]
    doc_type = str(args.DOC_TYPE).strip()

    try:
        config_path = find_config_path(args.config)
        extensions = parse_extensions(
            args.extensions,
            default=DEFAULT_EXTENSIONS,
        )
        if args.level_limit < 0:
            raise ValueError("--level-limit must be >= 0")
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)

    try:
        used = generate_document(
            doc_type=doc_type,
            output_path=output_path,
            inputs=input_paths,
            extensions=extensions,
            level_limit=int(args.level_limit),
            config_path=config_path,
        )
    except Exception as exc:
        print(f"Failed to generate document: {exc}")
        raise SystemExit(1)

    print(f"Generated document from {used} reference file(s) -> {output_path}")
    return 0


def _handle_config(argv: Sequence[str]) -> int:
    parser = _build_config_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return _handle_config_init(args)

    parser.error(f"Unsupported config command '{args.command}'.")
    return 2  # pragma: no cover - argparse.error exits before here


def _build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study generate-document config",
        description="Manage configuration files for generate-document.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Write the default documents.toml template.",
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

    try:
        template = config_templates.get_template(CONFIG_TEMPLATE_NAME)
    except ConfigTemplateError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    try:
        written = template.write(target, overwrite=args.force)
    except ConfigTemplateError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    sys.stdout.write(f"Wrote generate-document config to {written}\n")
    return 0


def _resolve_config_target(args: argparse.Namespace) -> Path:
    if args.path is not None:
        candidate = args.path.expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    layout = workspace_mod.ensure_workspace(path=args.workspace)
    return layout.path_for("config") / CONFIG_FILENAME
