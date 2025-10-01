"""Command-line interface for `study generate-document`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from study_utils.core import parse_extensions

from .config import find_config_path
from .runner import generate_document


DEFAULT_EXTENSIONS = {"txt", "md", "markdown"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown document from reference files using an AI "
            "prompt"
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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
