"""Utilities for resolving the Study RAG data directory layout.

All runtime artifacts (configs, vector DBs, sessions, logs) live under a
user-scoped data home. The default is ``~/.study-utils-data`` but callers can
override it with the ``STUDY_UTILS_DATA_HOME`` environment variable. Each
helper here is side-effect free unless ``create=True`` is passed, in which case
the required directories are created with user-only permissions when the
platform allows it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping


DATA_HOME_ENV = "STUDY_UTILS_DATA_HOME"
DEFAULT_DATA_HOME = Path.home() / ".study-utils-data"

_SUBDIRS = {
    "config": "config",
    "vector_dbs": "rag_dbs",
    "sessions": "rag_sessions",
    "logs": "logs",
}


class DataDirError(RuntimeError):
    """Raised when the data home cannot be resolved or created."""


def _coerce_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is None:
        return os.environ
    return env


def _chmod_safe(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except (NotImplementedError, PermissionError):
        # Some filesystems ignore chmod (e.g., Windows, mounted drives). Allow
        # execution to continue so callers still receive a usable directory.
        return


def _ensure_dir(path: Path, mode: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _chmod_safe(path, mode)
    return path


def _resolve_base(env: Mapping[str, str]) -> Path:
    override = env.get(DATA_HOME_ENV)
    if override is not None:
        override = override.strip()
    if override:
        base = Path(override).expanduser()
    else:
        base = DEFAULT_DATA_HOME
    try:
        return base.expanduser().resolve()
    except FileNotFoundError:
        # ``resolve`` on some systems requires parents to exist when strict.
        # Re-run without resolving parents by returning the absolute variant.
        return base.expanduser().absolute()


def _validate_dir_candidate(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise DataDirError(
            "Configured data home already exists and is not a directory: "
            f"{path}"
        )


def get_data_home(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    """Return the resolved data home directory, optionally creating it.

    Args:
        env: Optional mapping of environment variables. Defaults to
            ``os.environ`` when omitted to ease testing.
        create: When ``True`` (default), ensure the directory exists and apply
            secure permissions where possible.
    """

    env_map = _coerce_env(env)
    base = _resolve_base(env_map)
    _validate_dir_candidate(base)
    if create:
        _ensure_dir(base, 0o700)
        for key in _SUBDIRS.values():
            _ensure_dir(base / key, 0o700)
    return base


def require_subdir(
    name: str, *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    """Return a named subdirectory inside the data home.

    ``name`` must be one of the logical keys in ``_SUBDIRS``.
    """

    if name not in _SUBDIRS:
        raise KeyError(f"Unknown data subdir '{name}'.")
    base = get_data_home(env=env, create=create)
    target = base / _SUBDIRS[name]
    if create:
        _ensure_dir(target, 0o700)
    elif target.exists() and not target.is_dir():
        raise DataDirError(
            f"Expected a directory for '{name}' but found a file: {target}"
        )
    return target


def config_dir(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    return require_subdir("config", env=env, create=create)


def vector_db_dir(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    return require_subdir("vector_dbs", env=env, create=create)


def sessions_dir(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    return require_subdir("sessions", env=env, create=create)


def logs_dir(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    return require_subdir("logs", env=env, create=create)


def config_path(
    *, env: Mapping[str, str] | None = None, create_parent: bool = True
) -> Path:
    """Return the canonical config TOML path without creating the file."""

    cfg_dir = config_dir(env=env, create=create_parent)
    path = cfg_dir / "rag.toml"
    if path.exists() and path.is_dir():
        raise DataDirError(
            f"Configuration path exists but is a directory: {path}"
        )
    return path


def describe_layout(
    *, env: Mapping[str, str] | None = None
) -> MutableMapping[str, Path]:
    """Return a mapping describing the core directories for diagnostics."""

    base = get_data_home(env=env, create=False)
    return {
        "home": base,
        "config": base / _SUBDIRS["config"],
        "vector_dbs": base / _SUBDIRS["vector_dbs"],
        "sessions": base / _SUBDIRS["sessions"],
        "logs": base / _SUBDIRS["logs"],
    }
