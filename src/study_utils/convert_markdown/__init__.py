"""Public APIs for the document-to-Markdown converter scaffolding."""

from __future__ import annotations

from .converter import (
    ConversionError,
    ConversionOutcome,
    ConversionStatus,
    ConverterDependencies,
    DependencyError,
    UnsupportedFormatError,
    convert_file,
    SUPPORTED_EXTENSIONS,
)

from .executor import ExecutionSummary, run_conversion

from .config import (
    CollisionPolicy,
    ConfigOverrides,
    ConvertMarkdownConfig,
    ConvertMarkdownConfigError,
    LoadResult,
    load_config,
)

__all__ = [
    "ConversionError",
    "ConversionOutcome",
    "ConversionStatus",
    "ConverterDependencies",
    "DependencyError",
    "UnsupportedFormatError",
    "convert_file",
    "SUPPORTED_EXTENSIONS",
    "CollisionPolicy",
    "ConfigOverrides",
    "ConvertMarkdownConfig",
    "ConvertMarkdownConfigError",
    "LoadResult",
    "load_config",
    "ExecutionSummary",
    "run_conversion",
]
