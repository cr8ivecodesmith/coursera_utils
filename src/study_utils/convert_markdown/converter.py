"""Conversion pipeline for the document-to-Markdown workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from .config import CollisionPolicy
from .output import render_document

# Supported extensions mapped to the underlying conversion backend. The default
# config mirrors these values, but the pipeline keeps the authoritative list so
# callers can validate inputs before dispatching work.
_MARKITDOWN_EXTENSIONS: frozenset[str] = frozenset({
    "pdf",
    "docx",
    "html",
    "txt",
})
_EPUB_EXTENSIONS: frozenset[str] = frozenset({"epub"})

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"pdf", "docx", "html", "txt", "epub"}
)


class ConversionError(RuntimeError):
    """Raised when a document fails to convert."""


class UnsupportedFormatError(ConversionError):
    """Raised when the source file extension is not supported."""


class DependencyError(ConversionError):
    """Raised when required conversion dependencies are unavailable."""


class ConversionStatus(Enum):
    """Outcome status for a single conversion."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ConversionOutcome:
    """Result of converting (or attempting to convert) a single file."""

    source: Path
    status: ConversionStatus
    output_path: Optional[Path] = None
    reason: Optional[str] = None
    error: Optional[Exception] = None


@dataclass(frozen=True)
class ConverterDependencies:
    """Callable seams for backend-specific conversion logic."""

    markitdown: Callable[[Path], str]
    epub: Callable[[Path], str]


def convert_file(
    source: Path,
    *,
    output_dir: Path,
    collision: CollisionPolicy,
    dependencies: ConverterDependencies,
    now: Callable[[], datetime] | None = None,
) -> ConversionOutcome:
    """Convert ``source`` into Markdown according to ``collision`` policy."""

    try:
        return _convert_file(
            source,
            output_dir=output_dir,
            collision=collision,
            dependencies=dependencies,
            now=now or _default_now,
        )
    except Exception as exc:
        return ConversionOutcome(
            source=source,
            status=ConversionStatus.FAILED,
            reason=str(exc),
            error=exc,
        )


def _convert_file(
    source: Path,
    *,
    output_dir: Path,
    collision: CollisionPolicy,
    dependencies: ConverterDependencies,
    now: Callable[[], datetime],
) -> ConversionOutcome:
    normalized_source = source.resolve()
    if not normalized_source.exists():
        raise ConversionError(f"Source file not found: {normalized_source}")
    if not normalized_source.is_file():
        raise ConversionError(
            f"Source path is not a file: {normalized_source}"
        )

    extension = _normalize_extension(normalized_source)
    if extension in _MARKITDOWN_EXTENSIONS:
        body = dependencies.markitdown(normalized_source)
    elif extension in _EPUB_EXTENSIONS:
        body = dependencies.epub(normalized_source)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file extension '.{extension}' for conversion."
        )

    converted_at = _normalize_timestamp(now())
    source_mtime = _source_mtime(normalized_source)

    metadata = {
        "source_path": str(normalized_source),
        "converted_at": converted_at,
        "source_modified_at": source_mtime,
    }

    base_output = output_dir / f"{normalized_source.stem}.md"
    target_path, status, reason = _resolve_output_path(
        base_output,
        collision=collision,
    )
    if status is ConversionStatus.SKIPPED:
        return ConversionOutcome(
            source=normalized_source,
            status=status,
            output_path=target_path,
            reason=reason,
        )

    document = render_document(metadata, body)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(document, encoding="utf-8")

    return ConversionOutcome(
        source=normalized_source,
        status=ConversionStatus.SUCCESS,
        output_path=target_path,
    )


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_extension(path: Path) -> str:
    suffix = path.suffix
    if not suffix:
        raise UnsupportedFormatError(
            "Files without an extension are not supported."
        )
    return suffix.lstrip(".").lower()


def _normalize_timestamp(candidate: datetime) -> datetime:
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _source_mtime(path: Path) -> datetime:
    stat_result = path.stat()
    return datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)


def _resolve_output_path(
    base: Path,
    *,
    collision: CollisionPolicy,
) -> tuple[Path, Optional[ConversionStatus], Optional[str]]:
    if not base.exists():
        return base, None, None

    if collision is CollisionPolicy.SKIP:
        reason = (
            "Output already exists and collision policy is 'skip'."
        )
        return base, ConversionStatus.SKIPPED, reason

    if collision is CollisionPolicy.OVERWRITE:
        return base, None, None

    if collision is not CollisionPolicy.VERSION:
        policy_value = getattr(collision, "value", str(collision))
        raise ConversionError(
            f"Unsupported collision policy: {policy_value}"
        )

    counter = 1
    while True:
        candidate = base.with_name(
            f"{base.stem}-{counter:02d}{base.suffix}"
        )
        if not candidate.exists():
            return candidate, None, None
        counter += 1


__all__ = [
    "ConversionError",
    "UnsupportedFormatError",
    "DependencyError",
    "ConversionStatus",
    "ConversionOutcome",
    "ConverterDependencies",
    "convert_file",
    "SUPPORTED_EXTENSIONS",
]
