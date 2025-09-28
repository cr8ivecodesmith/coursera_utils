"""Sequential executor for convert-markdown runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import ConvertMarkdownConfig
from .converter import (
    ConversionOutcome,
    ConversionStatus,
    ConverterDependencies,
    convert_file,
)


@dataclass(frozen=True)
class ExecutionSummary:
    """Aggregated results for a convert-markdown run."""

    requested: tuple[Path, ...]
    processed: tuple[Path, ...]
    outcomes: tuple[ConversionOutcome, ...]

    @property
    def success_count(self) -> int:
        return sum(1 for outcome in self.outcomes if _is_success(outcome))

    @property
    def skipped_count(self) -> int:
        return sum(1 for outcome in self.outcomes if _is_skipped(outcome))

    @property
    def failure_count(self) -> int:
        return sum(1 for outcome in self.outcomes if _is_failure(outcome))

    @property
    def exit_code(self) -> int:
        return 1 if self.failure_count else 0


def run_conversion(
    inputs: Sequence[Path],
    *,
    config: ConvertMarkdownConfig,
    dependencies: ConverterDependencies,
    logger: logging.Logger,
) -> ExecutionSummary:
    """Process ``inputs`` sequentially and return an aggregated summary."""

    normalized_inputs = tuple(_normalize_inputs(inputs))
    configured_extensions = {ext.lower() for ext in config.extensions}

    logger.info(
        "Starting convert-markdown run",
        extra={
            "input_count": len(normalized_inputs),
            "extensions": sorted(configured_extensions),
            "output_dir": str(config.output_dir),
        },
    )

    candidates = tuple(_expand_inputs(normalized_inputs, configured_extensions))

    logger.info(
        "Prepared conversion candidates",
        extra={
            "candidate_count": len(candidates),
        },
    )

    outcomes: list[ConversionOutcome] = []
    for source in candidates:
        extension = _extension_for(source)
        if extension is not None and extension not in configured_extensions:
            outcome = ConversionOutcome(
                source=source,
                status=ConversionStatus.SKIPPED,
                reason=(
                    "Extension '.{0}' not enabled in configuration.".format(
                        extension
                    )
                ),
            )
            outcomes.append(outcome)
            logger.info(
                "Skipped source due to extension filter",
                extra={"source": str(source), "extension": extension},
            )
            continue

        outcome = convert_file(
            source,
            output_dir=config.output_dir,
            collision=config.collision,
            dependencies=dependencies,
        )
        outcomes.append(outcome)

        if outcome.status is ConversionStatus.SUCCESS:
            output_path_value = None
            if outcome.output_path is not None:
                output_path_value = str(outcome.output_path)
            logger.info(
                "Converted document",
                extra={
                    "source": str(outcome.source),
                    "output_path": output_path_value,
                },
            )
        elif outcome.status is ConversionStatus.SKIPPED:
            logger.info(
                "Skipped document",
                extra={"source": str(outcome.source), "reason": outcome.reason},
            )
        else:
            logger.error(
                "Failed to convert document",
                extra={
                    "source": str(outcome.source),
                    "reason": outcome.reason,
                },
            )

    summary = ExecutionSummary(
        requested=normalized_inputs,
        processed=candidates,
        outcomes=tuple(outcomes),
    )

    logger.info(
        "Completed convert-markdown run",
        extra={
            "success_count": summary.success_count,
            "skipped_count": summary.skipped_count,
            "failure_count": summary.failure_count,
        },
    )

    return summary


def _normalize_inputs(inputs: Sequence[Path]) -> Iterable[Path]:
    for raw in inputs:
        normalized = raw.expanduser()
        try:
            yield normalized.resolve(strict=False)
        except FileNotFoundError:
            # resolve(strict=False) may still raise on certain platforms if the
            # parent directory is missing; fall back to the expanded path.
            yield normalized


def _expand_inputs(
    inputs: Sequence[Path],
    configured_extensions: set[str],
) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in sorted(inputs, key=lambda candidate: str(candidate)):
        if path.is_dir():
            for child in _iter_directory(path, configured_extensions):
                if child not in seen:
                    seen.add(child)
                    yield child
        else:
            if path not in seen:
                seen.add(path)
                yield path


def _iter_directory(directory: Path, extensions: set[str]) -> Iterable[Path]:
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        extension = _extension_for(candidate)
        if extension is None:
            continue
        if extension in extensions:
            yield candidate


def _extension_for(path: Path) -> str | None:
    suffix = path.suffix
    if not suffix:
        return None
    return suffix.lstrip(".").lower()


def _is_success(outcome: ConversionOutcome) -> bool:
    return outcome.status is ConversionStatus.SUCCESS


def _is_skipped(outcome: ConversionOutcome) -> bool:
    return outcome.status is ConversionStatus.SKIPPED


def _is_failure(outcome: ConversionOutcome) -> bool:
    return outcome.status is ConversionStatus.FAILED


__all__ = [
    "ExecutionSummary",
    "run_conversion",
]
