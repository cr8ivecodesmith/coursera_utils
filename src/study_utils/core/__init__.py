"""Core shared helpers for study_utils subcommands."""

from __future__ import annotations

from .ai import load_client
from .config import (
    TomlConfigError,
    load_toml,
    merge_defaults,
    write_toml_template,
)
from .files import (
    parse_extensions,
    iter_text_files,
    order_files,
    read_text_file,
)
from .logging import JsonLogFormatter, configure_logger

__all__ = [
    "load_client",
    "TomlConfigError",
    "load_toml",
    "merge_defaults",
    "write_toml_template",
    "parse_extensions",
    "iter_text_files",
    "order_files",
    "read_text_file",
    "configure_logger",
    "JsonLogFormatter",
]
