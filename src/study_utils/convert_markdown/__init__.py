"""Public APIs for the document-to-Markdown converter scaffolding."""

from __future__ import annotations

from .config import (
    CollisionPolicy,
    ConfigOverrides,
    ConvertMarkdownConfig,
    ConvertMarkdownConfigError,
    LoadResult,
    load_config,
)

__all__ = [
    "CollisionPolicy",
    "ConfigOverrides",
    "ConvertMarkdownConfig",
    "ConvertMarkdownConfigError",
    "LoadResult",
    "load_config",
]
