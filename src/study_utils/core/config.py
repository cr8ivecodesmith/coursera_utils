"""Shared TOML configuration helpers for study_utils commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:  # Python >= 3.11 ships ``tomllib`` in the stdlib.
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError("Python 3.11+ is required for tomllib support.") from exc

__all__ = [
    "TomlConfigError",
    "load_toml",
    "merge_defaults",
    "write_toml_template",
]


class TomlConfigError(RuntimeError):
    """Raised when TOML config IO or validation fails."""


def load_toml(path: Path) -> Mapping[str, Any]:
    """Load a TOML document from ``path``.

    Errors are surfaced as :class:`TomlConfigError` instances so callers can
    translate them into domain-specific exceptions.
    """

    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise TomlConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TomlConfigError(f"Failed to parse config TOML: {exc}") from exc


def merge_defaults(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
    *,
    path: str = "",
) -> None:
    """Recursively merge ``override`` into ``base`` enforcing known keys."""

    for key, value in override.items():
        if key not in base:
            dotted = f"{path}{key}" if path else key
            raise TomlConfigError(f"Unknown configuration key '{dotted}'.")
        base_value = base[key]
        if isinstance(base_value, MutableMapping):
            if not isinstance(value, Mapping):
                dotted = f"{path}{key}" if path else key
                raise TomlConfigError(
                    "Expected table for '{0}', found {1}.".format(
                        dotted,
                        type(value).__name__,
                    )
                )
            merge_defaults(base_value, value, path=f"{path}{key}.")
            continue
        base[key] = value


def write_toml_template(
    path: Path,
    *,
    template: str,
    overwrite: bool = False,
    mode: int = 0o600,
) -> Path:
    """Write ``template`` to ``path`` honouring ``overwrite`` semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise TomlConfigError(f"Config already exists: {path}")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(template)
    try:
        path.chmod(mode)
    except PermissionError:  # pragma: no cover - depends on filesystem
        pass
    return path
