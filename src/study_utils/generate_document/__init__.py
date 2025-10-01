"""Generate document package exposing CLI, config, and runner helpers."""

from .cli import build_arg_parser, main
from .config import GenerateOptions, find_config_path, load_documents_config
from .runner import build_messages, build_reference_block, generate_document

__all__ = [
    "GenerateOptions",
    "build_arg_parser",
    "build_messages",
    "build_reference_block",
    "find_config_path",
    "generate_document",
    "load_documents_config",
    "main",
]
