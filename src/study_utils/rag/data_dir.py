"""Utilities for resolving the Study RAG data directory layout.

All runtime artifacts (configs, vector DBs, sessions, logs) live under the
shared study-utils workspace. The workspace helper lives in
``study_utils.core.workspace``; this module keeps the existing API surface for
RAG-specific callers while delegating to the shared implementation so that
other tools (e.g., convert-markdown) operate over the same directory tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

from study_utils.core import workspace


DATA_HOME_ENV = workspace.WORKSPACE_ENV
DEFAULT_DATA_HOME = Path.home() / ".study-utils-data"

_SUBDIRS = {
    "config": "config",
    "vector_dbs": "rag_dbs",
    "sessions": "rag_sessions",
    "logs": "logs",
}


class DataDirError(RuntimeError):
    """Raised when the data home cannot be resolved or created."""


def _resolve_layout(
    *, env: Mapping[str, str] | None, create: bool
) -> workspace.WorkspaceLayout:
    try:
        if env is None:
            return workspace.ensure_workspace(create=create, subdirs=_SUBDIRS)

        override = env.get(DATA_HOME_ENV)
        if override is not None:
            override = override.strip()
        base_path = (
            Path(override).expanduser() if override else DEFAULT_DATA_HOME
        )
        return workspace.ensure_workspace(
            path=base_path,
            create=create,
            subdirs=_SUBDIRS,
        )
    except workspace.WorkspaceError as exc:  # pragma: no cover - passthrough
        raise DataDirError(str(exc)) from exc


def get_data_home(
    *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    """Return the resolved data home directory, optionally creating it."""

    layout = _resolve_layout(env=env, create=create)
    return layout.home


def require_subdir(
    name: str, *, env: Mapping[str, str] | None = None, create: bool = True
) -> Path:
    """Return a named subdirectory inside the data home."""

    try:
        _ = _SUBDIRS[name]
    except KeyError as exc:  # pragma: no cover - guard for misuse
        raise KeyError(f"Unknown data subdir '{name}'.") from exc

    layout = _resolve_layout(env=env, create=create)
    target = layout.path_for(name)
    if not create and target.exists() and not target.is_dir():
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
