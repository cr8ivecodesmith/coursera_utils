"""Common file handling utilities shared across study_utils modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set

__all__ = [
    "parse_extensions",
    "iter_text_files",
    "order_files",
    "read_text_file",
]


def parse_extensions(
    values: Optional[Sequence[str]],
    *,
    default: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Normalize extension strings to a lowercase set without leading dots.

    Parameters
    ----------
    values:
        Raw extension inputs (with or without leading dots).
        ``None`` returns the provided default set.
    default:
        Fallback extensions when ``values`` is empty. Defaults to ``{"txt"}``.
    """
    fallback = set(default or {"txt"})
    if not values:
        return set(fallback)

    normalized: Set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower()
        if candidate.startswith("."):
            candidate = candidate[1:]
        if candidate:
            normalized.add(candidate)
    return normalized or set(fallback)


def iter_text_files(
    paths: Sequence[Path],
    extensions: Set[str],
    level_limit: int,
) -> Iterator[Path]:
    """Yield matching files from the given paths, preserving input order."""
    if level_limit < 0:
        raise ValueError("level_limit must be >= 0")

    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if _matches_extension(path, extensions):
                yield path
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        if not path.is_dir():
            continue
        yield from _iter_text_directory(path, extensions, level_limit)


def _iter_text_directory(
    root: Path, extensions: Set[str], level_limit: int
) -> Iterator[Path]:
    for candidate in _sorted_directory_files(root):
        if level_limit and not _within_level_limit(
            candidate, root, level_limit
        ):
            continue
        if _matches_extension(candidate, extensions):
            yield candidate


def _sorted_directory_files(root: Path) -> List[Path]:
    return sorted(
        (child for child in root.rglob("*") if child.is_file()),
        key=lambda p: p.name.lower(),
    )


def _within_level_limit(path: Path, root: Path, level_limit: int) -> bool:
    try:
        rel = path.relative_to(root)
    except Exception:
        return False
    return len(rel.parts) <= level_limit


def _matches_extension(path: Path, extensions: Set[str]) -> bool:
    return path.is_file() and path.suffix.lower().lstrip(".") in extensions


def order_files(files: Sequence[Path], order_by: Optional[str]) -> List[Path]:
    """Order files by the requested attribute, preserving input on ``None``."""
    if not order_by:
        return list(files)

    reverse = order_by.startswith("-")
    key = order_by.lstrip("-")

    def keyfn(p: Path):
        if key == "name":
            return p.name.lower()
        try:
            stats = p.stat()
        except Exception:
            return float("inf")
        if key == "created":
            return getattr(stats, "st_birthtime", None) or stats.st_ctime
        if key == "modified":
            return stats.st_mtime
        return 0

    return sorted(files, key=keyfn, reverse=reverse)


def read_text_file(path: Path) -> str:
    """Read a text file as UTF-8 with replacement for decode errors."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
