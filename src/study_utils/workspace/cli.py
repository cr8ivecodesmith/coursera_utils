"""CLI entry points for shared workspace management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

from study_utils.core import workspace as workspace_mod


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study init",
        description=(
            "Bootstrap the shared study-utils workspace and ensure required "
            "subdirectories exist."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        help=(
            "Override the workspace root (defaults to STUDY_UTILS_DATA_HOME "
            "or ~/.study-utils-data)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output on success.",
    )
    return parser


def _format_created(created: Mapping[str, bool], key: str) -> str:
    return "created" if created.get(key, False) else "exists"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    layout = workspace_mod.ensure_workspace(path=args.path)

    if args.quiet:
        return 0

    created = layout.created
    home_status = _format_created(created, "home")
    lines = [f"Workspace ready at {layout.home} ({home_status})"]

    if layout.directories:
        lines.append("Subdirectories:")
        width = max(len(name) for name in layout.directories)
        for name, directory in layout.items():
            status = _format_created(created, name)
            lines.append(f"  {name.ljust(width)}  {directory} ({status})")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())
